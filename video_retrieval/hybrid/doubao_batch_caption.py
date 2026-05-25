from __future__ import annotations

import argparse
import base64
import json
import os
import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("OPENCV_FFMPEG_DEBUG", "0")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "8")

import cv2
import numpy as np
from openai import OpenAI

from ..env_loader import load_default_dotenv_files
from ..io_utils import write_json
from .config import DEFAULT_VIDEO_DIR
from .metadata_store import MetadataStore
from ..clip.segmenter import iter_videos
from ..portable_paths import portable_path_text
from ..profile_paths import default_hybrid_captions_jsonl, default_hybrid_metadata_db_path, resolve_path
from .search_text import build_caption_term_text, build_description_term_text, build_tag_term_text

DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_API_KEY_ENV = "ARK_API_KEY"
DEFAULT_MODEL_ENV = "DOUBAO_MODEL"
LEGACY_MODEL_ENV = "DOUBAO_ENDPOINT_ID"
DEFAULT_OUTPUT_PATH = default_hybrid_captions_jsonl(None)
DEFAULT_PROMPT = (
    "你将看到按时间顺序抽取的一组视频关键帧。"
    "请基于画面内容，为视频生成适合检索的结构化文本。"
    "标签需覆盖核心实体、关键动作、主要场景；"
    "详细描述需用一句简洁、客观的简体中文概括视频内容。"
    "不要猜测声音、对白、情绪、身份或画面外信息。"
)
STRUCTURED_OUTPUT_INSTRUCTIONS = (
    "请只输出一个 JSON 对象，不要输出 Markdown，不要输出额外解释。"
    ' JSON 格式必须为 {"tags":["标签1","标签2"],"description":"一句话描述"}。'
    "其中 tags 是 3 到 8 个简短中文标签，不重复；"
    "要优先包含实体、动作、场景三类信息。"
    "要求 description 只有一句话，客观、简洁，适合视频检索入库。"
)
DEFAULT_SCENE_SCAN_STRIDE_SECONDS = 1.0
DEFAULT_SCENE_CHANGE_THRESHOLD = 0.12
DEFAULT_PROGRESS_INTERVAL_SECONDS = 5.0
CV2_LOG_LEVELS = {
    "SILENT": 0,
    "FATAL": 1,
    "ERROR": 2,
    "WARNING": 3,
    "WARN": 3,
    "INFO": 4,
    "DEBUG": 5,
    "VERBOSE": 6,
}

_THREAD_STATE = threading.local()
_WRITE_LOCK = threading.Lock()


@dataclass
class CaptionConfig:
    video_dir: Path
    output_jsonl: Path
    output_json: Path | None
    metadata_db_path: Path | None
    sync_metadata_db: bool
    video_ids: tuple[str, ...]
    video_index_start: int | None
    video_index_end: int | None
    model: str
    api_key: str
    base_url: str
    api_mode: str
    limit: int
    workers: int
    sample_frames: int
    max_side: int
    jpg_quality: int
    temperature: float
    max_tokens: int
    timeout_seconds: float
    prompt: str
    overwrite: bool
    retry_failed: bool
    scene_scan_stride_seconds: float
    scene_change_threshold: float
    progress_interval_seconds: float


@dataclass
class SampledFrame:
    timestamp_seconds: float
    data_url: str


@dataclass
class StructuredCaption:
    tags: list[str]
    description: str


def configure_opencv_logging():
    set_log_level = getattr(cv2, "setLogLevel", None)
    if not callable(set_log_level):
        return
    raw_level = str(os.environ.get("OPENCV_LOG_LEVEL", "SILENT")).strip()
    if not raw_level:
        return
    if raw_level.lstrip("-").isdigit():
        level = int(raw_level)
    else:
        level = CV2_LOG_LEVELS.get(raw_level.upper())
    if level is None:
        return
    try:
        set_log_level(level)
    except Exception:
        return


configure_opencv_logging()


def parse_args():
    parser = argparse.ArgumentParser(description="Batch caption videos with Doubao multimodal API.")
    parser.add_argument("--video-dir", default=str(DEFAULT_VIDEO_DIR))
    parser.add_argument("--output-jsonl", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--output-json", help="Optional aggregated JSON output keyed by video_id.")
    parser.add_argument(
        "--profile", help="Named storage profile for syncing structured captions into metadata.db."
    )
    parser.add_argument(
        "--metadata-db",
        help="Optional metadata.db path for syncing tags/description into retrieval database.",
    )
    parser.add_argument(
        "--no-sync-metadata-db",
        action="store_true",
        help="Disable writing structured captions into metadata.db.",
    )
    parser.add_argument(
        "--video-id",
        dest="video_ids",
        action="append",
        help="Only process the specified video_id (for example: video2999). Repeat to pass multiple ids.",
    )
    parser.add_argument(
        "--video-index-start",
        type=int,
        help="Only process videos whose numeric id is >= this value.",
    )
    parser.add_argument(
        "--video-index-end", type=int, help="Only process videos whose numeric id is <= this value."
    )
    parser.add_argument("--model", help=f"Doubao model name. Defaults to {DEFAULT_MODEL_ENV}.")
    parser.add_argument("--endpoint-id", dest="legacy_model", help="Deprecated alias of --model.")
    parser.add_argument(
        "--api-key-env", default=DEFAULT_API_KEY_ENV, help="Env var name for the Ark API key."
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-mode", choices=("responses", "chat"), default="chat")
    parser.add_argument(
        "--limit", type=int, default=0, help="Only process the first N videos. 0 means all."
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--sample-frames", type=int, default=8)
    parser.add_argument("--max-side", type=int, default=640)
    parser.add_argument("--jpg-quality", type=int, default=85)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=180)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument(
        "--prompt-file", help="Optional text file. Non-empty lines are joined with spaces."
    )
    parser.add_argument(
        "--scene-scan-stride-seconds", type=float, default=DEFAULT_SCENE_SCAN_STRIDE_SECONDS
    )
    parser.add_argument(
        "--scene-change-threshold", type=float, default=DEFAULT_SCENE_CHANGE_THRESHOLD
    )
    parser.add_argument(
        "--progress-interval-seconds",
        type=float,
        default=DEFAULT_PROGRESS_INTERVAL_SECONDS,
        help="How often to print heartbeat progress while waiting for workers.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    return parser.parse_args()


def load_prompt(path_value: str | None) -> str:
    if not path_value:
        return DEFAULT_PROMPT
    path = Path(path_value)
    text = path.read_text(encoding="utf-8")
    parts = [line.strip() for line in text.splitlines() if line.strip()]
    return " ".join(parts) if parts else DEFAULT_PROMPT


def build_structured_prompt(prompt: str) -> str:
    return f"{prompt}\n{STRUCTURED_OUTPUT_INSTRUCTIONS}"


def build_config(args) -> CaptionConfig:
    load_default_dotenv_files()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key environment variable: {args.api_key_env}")
    model = (
        args.model
        or args.legacy_model
        or os.environ.get(DEFAULT_MODEL_ENV)
        or os.environ.get(LEGACY_MODEL_ENV)
    )
    if not model:
        raise RuntimeError(
            f"Missing model. Pass --model or set {DEFAULT_MODEL_ENV}. "
            f"(Compatible fallback: {LEGACY_MODEL_ENV})"
        )
    sync_metadata_db = not args.no_sync_metadata_db
    metadata_db_path = (
        resolve_path(args.metadata_db, default_hybrid_metadata_db_path(args.profile))
        if sync_metadata_db
        else None
    )
    return CaptionConfig(
        video_dir=Path(args.video_dir),
        output_jsonl=Path(args.output_jsonl),
        output_json=Path(args.output_json) if args.output_json else None,
        metadata_db_path=metadata_db_path,
        sync_metadata_db=sync_metadata_db,
        video_ids=tuple(args.video_ids or ()),
        video_index_start=args.video_index_start,
        video_index_end=args.video_index_end,
        model=model,
        api_key=api_key,
        base_url=args.base_url,
        api_mode=args.api_mode,
        limit=args.limit,
        workers=max(1, args.workers),
        sample_frames=max(1, args.sample_frames),
        max_side=max(128, args.max_side),
        jpg_quality=max(50, min(100, args.jpg_quality)),
        temperature=args.temperature,
        max_tokens=max(32, args.max_tokens),
        timeout_seconds=max(10.0, args.timeout_seconds),
        prompt=load_prompt(args.prompt_file),
        overwrite=args.overwrite,
        retry_failed=args.retry_failed,
        scene_scan_stride_seconds=max(0.25, args.scene_scan_stride_seconds),
        scene_change_threshold=max(0.01, min(1.0, args.scene_change_threshold)),
        progress_interval_seconds=max(1.0, args.progress_interval_seconds),
    )


def get_client(config: CaptionConfig) -> OpenAI:
    client = getattr(_THREAD_STATE, "client", None)
    client_key = getattr(_THREAD_STATE, "client_key", None)
    expected_key = (config.base_url, config.api_key, config.timeout_seconds)
    if client is None or client_key != expected_key:
        client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout_seconds,
        )
        _THREAD_STATE.client = client
        _THREAD_STATE.client_key = expected_key
    return client


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def read_existing_results(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    rows: dict[str, dict] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        video_id = payload.get("video_id")
        if video_id:
            rows[video_id] = payload
    return rows


def extract_video_index(video_id: str) -> int | None:
    match = re.search(r"(\d+)$", video_id)
    if match is None:
        return None
    return int(match.group(1))


def append_jsonl(path: Path, payload: dict):
    ensure_parent(path)
    text = json.dumps(payload, ensure_ascii=False)
    with _WRITE_LOCK, path.open("a", encoding="utf-8") as handle:
        handle.write(text + "\n")


def rewrite_jsonl(path: Path, rows: list[dict]):
    ensure_parent(path)
    with _WRITE_LOCK, path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def resize_frame(frame: np.ndarray, max_side: int) -> np.ndarray:
    height, width = frame.shape[:2]
    longest = max(width, height)
    if longest <= max_side:
        return frame
    scale = max_side / float(longest)
    resized = cv2.resize(
        frame,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized


def encode_jpg_data_url(frame: np.ndarray, jpg_quality: int) -> str:
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])
    if not ok:
        raise RuntimeError("Failed to encode sampled frame as jpg.")
    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def uniform_sample_indices(total_frames: int, sample_frames: int) -> list[int]:
    if total_frames <= 0:
        return []
    if total_frames <= sample_frames:
        return list(range(total_frames))
    values = np.linspace(0, total_frames - 1, num=sample_frames)
    indices = []
    seen = set()
    for value in values:
        idx = int(round(float(value)))
        if idx not in seen:
            indices.append(idx)
            seen.add(idx)
    if len(indices) < sample_frames:
        for idx in range(total_frames):
            if idx not in seen:
                indices.append(idx)
                seen.add(idx)
            if len(indices) >= sample_frames:
                break
    return indices[:sample_frames]


def downsample_sorted_indices(indices: list[int], limit: int) -> list[int]:
    if limit <= 0 or not indices:
        return []
    unique_sorted = sorted(set(indices))
    if len(unique_sorted) <= limit:
        return unique_sorted
    positions = np.linspace(0, len(unique_sorted) - 1, num=limit)
    selected = []
    seen = set()
    for value in positions:
        idx = unique_sorted[int(round(float(value)))]
        if idx not in seen:
            selected.append(idx)
            seen.add(idx)
    if len(selected) < limit:
        for idx in unique_sorted:
            if idx not in seen:
                selected.append(idx)
                seen.add(idx)
            if len(selected) >= limit:
                break
    return sorted(selected[:limit])


def compute_scene_signature(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)


def detect_scene_change_indices(
    video_path: Path,
    total_frames: int,
    fps: float,
    sample_frames: int,
    scan_stride_seconds: float,
    scene_change_threshold: float,
) -> list[int]:
    if total_frames <= 0 or fps <= 0 or sample_frames <= 1:
        return []
    stride_frames = max(1, int(round(fps * scan_stride_seconds)))
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return []
    try:
        scene_indices: list[int] = []
        previous_signature: np.ndarray | None = None
        max_candidates = max(sample_frames * 4, sample_frames + 2)
        for frame_idx in range(0, total_frames, stride_frames):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            signature = compute_scene_signature(frame)
            if previous_signature is None:
                scene_indices.append(frame_idx)
                previous_signature = signature
                continue
            diff_score = float(np.mean(cv2.absdiff(signature, previous_signature)) / 255.0)
            if diff_score >= scene_change_threshold:
                scene_indices.append(frame_idx)
                previous_signature = signature
            if len(scene_indices) >= max_candidates:
                break
        return downsample_sorted_indices(scene_indices, max(sample_frames, len(scene_indices)))
    finally:
        capture.release()


def choose_keyframe_indices(
    video_path: Path,
    total_frames: int,
    fps: float,
    sample_frames: int,
    scan_stride_seconds: float,
    scene_change_threshold: float,
) -> list[int]:
    if total_frames <= 0:
        return []
    if sample_frames <= 1:
        return [0]
    timeline_count = max(1, min(sample_frames, (sample_frames + 1) // 2))
    timeline_indices = uniform_sample_indices(total_frames, timeline_count)
    scene_indices = detect_scene_change_indices(
        video_path=video_path,
        total_frames=total_frames,
        fps=fps,
        sample_frames=sample_frames,
        scan_stride_seconds=scan_stride_seconds,
        scene_change_threshold=scene_change_threshold,
    )
    if not scene_indices:
        return uniform_sample_indices(total_frames, sample_frames)
    remaining_slots = max(0, sample_frames - len(timeline_indices))
    timeline_index_set = set(timeline_indices)
    extra_scene_indices = [idx for idx in scene_indices if idx not in timeline_index_set]
    selected_scene_indices = downsample_sorted_indices(extra_scene_indices, remaining_slots)
    merged = sorted(set(timeline_indices + selected_scene_indices))
    if len(merged) < sample_frames:
        merged = downsample_sorted_indices(
            merged + uniform_sample_indices(total_frames, sample_frames), sample_frames
        )
    return downsample_sorted_indices(merged, sample_frames)


def sample_video_frames_fallback(
    capture: cv2.VideoCapture, sample_frames: int, max_side: int, jpg_quality: int
) -> tuple[list[SampledFrame], float]:
    sampled: list[SampledFrame] = []
    frame_idx = -1
    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    step = 1
    while len(sampled) < sample_frames:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        frame_idx += 1
        if frame_idx % step != 0:
            continue
        frame = resize_frame(frame, max_side)
        timestamp = float(frame_idx / fps) if fps > 0 else 0.0
        sampled.append(
            SampledFrame(
                timestamp_seconds=round(timestamp, 3),
                data_url=encode_jpg_data_url(frame, jpg_quality),
            )
        )
    return sampled, 0.0


def sample_video_frames(
    video_path: Path,
    sample_frames: int,
    max_side: int,
    jpg_quality: int,
    scan_stride_seconds: float,
    scene_change_threshold: float,
) -> tuple[list[SampledFrame], float]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = float(total_frames / fps) if fps > 0 and total_frames > 0 else 0.0
        if total_frames <= 0:
            sampled, fallback_duration = sample_video_frames_fallback(
                capture, sample_frames, max_side, jpg_quality
            )
            if not sampled:
                raise RuntimeError(f"Video has no readable frames: {video_path}")
            return sampled, fallback_duration
        indices = choose_keyframe_indices(
            video_path=video_path,
            total_frames=total_frames,
            fps=fps,
            sample_frames=sample_frames,
            scan_stride_seconds=scan_stride_seconds,
            scene_change_threshold=scene_change_threshold,
        )
        if not indices:
            raise RuntimeError(f"Video has no readable frames: {video_path}")

        sampled: list[SampledFrame] = []
        for frame_idx in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            frame = resize_frame(frame, max_side)
            timestamp = float(frame_idx / fps) if fps > 0 else 0.0
            sampled.append(
                SampledFrame(
                    timestamp_seconds=round(timestamp, 3),
                    data_url=encode_jpg_data_url(frame, jpg_quality),
                )
            )
        if not sampled:
            raise RuntimeError(f"Failed to sample readable frames: {video_path}")
        return sampled, duration
    finally:
        capture.release()


def build_messages(prompt: str, frames: list[SampledFrame]) -> list[dict]:
    content: list[dict] = [{"type": "text", "text": prompt}]
    for index, frame in enumerate(frames, start=1):
        content.append(
            {
                "type": "text",
                "text": f"第{index}帧，时间 {frame.timestamp_seconds:.3f} 秒。",
            }
        )
        content.append({"type": "image_url", "image_url": {"url": frame.data_url}})
    return [{"role": "user", "content": content}]


def build_response_input(prompt: str, frames: list[SampledFrame]) -> list[dict]:
    content: list[dict] = [{"type": "input_text", "text": prompt}]
    for index, frame in enumerate(frames, start=1):
        content.append(
            {
                "type": "input_text",
                "text": f"第{index}帧，时间 {frame.timestamp_seconds:.3f} 秒。",
            }
        )
        content.append({"type": "input_image", "image_url": frame.data_url})
    return [{"role": "user", "content": content}]


def extract_text_from_chat_response(response) -> str:
    choice = response.choices[0]
    message = getattr(choice, "message", None)
    if message is None:
        return ""
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            else:
                text = getattr(item, "text", None)
                if text:
                    parts.append(str(text))
        return "\n".join(part.strip() for part in parts if part and part.strip()).strip()
    return str(content).strip()


def extract_text_from_responses_api(response) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text).strip()

    output = getattr(response, "output", None) or []
    parts: list[str] = []
    for item in output:
        content_items = getattr(item, "content", None)
        if content_items is None and isinstance(item, dict):
            content_items = item.get("content", [])
        for content_item in content_items or []:
            if isinstance(content_item, dict):
                text = content_item.get("text")
            else:
                text = getattr(content_item, "text", None)
            if text:
                parts.append(str(text).strip())
    return "\n".join(part for part in parts if part).strip()


def strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def extract_first_json_object(text: str) -> str | None:
    cleaned = strip_code_fence(text)
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    return match.group(0).strip() if match else None


def normalize_tags(value) -> list[str]:
    if isinstance(value, str):
        raw_tags = re.split(r"[,，;；、|\n]+", value)
    elif isinstance(value, list):
        raw_tags = [str(item) for item in value]
    else:
        raw_tags = []
    tags: list[str] = []
    seen = set()
    for raw_tag in raw_tags:
        tag = str(raw_tag).strip()
        if not tag or tag in seen:
            continue
        tags.append(tag)
        seen.add(tag)
    return tags


def parse_structured_caption(text: str) -> StructuredCaption:
    json_text = extract_first_json_object(text)
    if json_text:
        try:
            payload = json.loads(json_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Caption API returned invalid JSON: {exc}") from exc
    else:
        payload = {}
    tags = normalize_tags(payload.get("tags") or payload.get("Tags") or payload.get("标签"))
    description = str(
        payload.get("description")
        or payload.get("Description")
        or payload.get("描述")
        or payload.get("caption")
        or ""
    ).strip()
    if not description:
        fallback_text = strip_code_fence(text)
        if fallback_text and not json_text:
            description = fallback_text
    if not description:
        raise RuntimeError("Caption API returned empty structured description.")
    if not tags:
        raise RuntimeError("Caption API returned no structured tags.")
    return StructuredCaption(tags=tags[:8], description=description)


def structured_caption_to_payload(caption: StructuredCaption) -> dict:
    return {
        "tags": caption.tags,
        "description": caption.description,
    }


def is_nonempty_caption(text: str | None) -> bool:
    return bool((text or "").strip())


def request_caption_via_responses(
    config: CaptionConfig, frames: list[SampledFrame]
) -> StructuredCaption:
    client = get_client(config)
    response = client.responses.create(
        model=config.model,
        input=build_response_input(build_structured_prompt(config.prompt), frames),
        temperature=config.temperature,
        max_output_tokens=config.max_tokens,
    )
    caption_text = extract_text_from_responses_api(response)
    if is_nonempty_caption(caption_text):
        return parse_structured_caption(caption_text)

    status = getattr(response, "status", None)
    incomplete_details = getattr(response, "incomplete_details", None)
    if not isinstance(incomplete_details, dict) and incomplete_details is not None:
        incomplete_details = getattr(incomplete_details, "model_dump", lambda: None)() or {
            "reason": getattr(incomplete_details, "reason", None)
        }
    reason = (incomplete_details or {}).get("reason")
    raise RuntimeError(f"Responses API returned empty caption (status={status}, reason={reason})")


def request_caption_via_chat(
    config: CaptionConfig, frames: list[SampledFrame]
) -> StructuredCaption:
    client = get_client(config)
    response = client.chat.completions.create(
        model=config.model,
        messages=build_messages(build_structured_prompt(config.prompt), frames),
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    caption_text = extract_text_from_chat_response(response)
    if is_nonempty_caption(caption_text):
        return parse_structured_caption(caption_text)
    raise RuntimeError("Chat Completions API returned empty caption.")


def request_caption(config: CaptionConfig, frames: list[SampledFrame]) -> StructuredCaption:
    if config.api_mode == "responses":
        try:
            return request_caption_via_responses(config, frames)
        except Exception as exc:
            print(f"[warn] responses fallback to chat: {exc}", flush=True)
            return request_caption_via_chat(config, frames)
    return request_caption_via_chat(config, frames)


def describe_video(config: CaptionConfig, video_path: Path) -> dict:
    video_id = video_path.stem
    started_at = time.time()
    frames, duration = sample_video_frames(
        video_path=video_path,
        sample_frames=config.sample_frames,
        max_side=config.max_side,
        jpg_quality=config.jpg_quality,
        scan_stride_seconds=config.scene_scan_stride_seconds,
        scene_change_threshold=config.scene_change_threshold,
    )
    structured_caption = request_caption(config, frames)
    if not is_nonempty_caption(structured_caption.description):
        raise RuntimeError("Caption API returned empty text.")
    latency_seconds = round(time.time() - started_at, 3)
    return {
        "video_id": video_id,
        "video_path": portable_path_text(video_path) or str(video_path.resolve()),
        "status": "ok",
        "caption": structured_caption.description,
        "description": structured_caption.description,
        "tags": structured_caption.tags,
        "structured_caption": structured_caption_to_payload(structured_caption),
        "sampled_frames": len(frames),
        "duration_seconds": round(duration, 3),
        "model": config.model,
        "base_url": config.base_url,
        "latency_seconds": latency_seconds,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def should_skip(existing: dict | None, overwrite: bool, retry_failed: bool) -> bool:
    if existing is None:
        return False
    if overwrite:
        return False
    status = existing.get("status")
    description = existing.get("description") or existing.get("caption")
    if status == "ok" and is_nonempty_caption(description):
        return True
    return bool(status == "error" and not retry_failed)


def process_video(config: CaptionConfig, video_path: Path) -> dict:
    try:
        return describe_video(config, video_path)
    except Exception as exc:
        return {
            "video_id": video_path.stem,
            "video_path": portable_path_text(video_path) or str(video_path.resolve()),
            "status": "error",
            "caption": None,
            "description": None,
            "tags": [],
            "structured_caption": None,
            "model": config.model,
            "base_url": config.base_url,
            "error": str(exc),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }


def select_videos(config: CaptionConfig, existing_rows: dict[str, dict]) -> list[Path]:
    video_paths = list(iter_videos(config.video_dir))
    if config.video_ids:
        allowed = set(config.video_ids)
        video_paths = [path for path in video_paths if path.stem in allowed]
    if config.video_index_start is not None or config.video_index_end is not None:

        def in_index_range(path: Path) -> bool:
            index = extract_video_index(path.stem)
            if index is None:
                return False
            if config.video_index_start is not None and index < config.video_index_start:
                return False
            return not (config.video_index_end is not None and index > config.video_index_end)

        video_paths = [path for path in video_paths if in_index_range(path)]
    if config.limit > 0:
        video_paths = video_paths[: config.limit]
    pending = [
        path
        for path in video_paths
        if not should_skip(existing_rows.get(path.stem), config.overwrite, config.retry_failed)
    ]
    return pending


def print_progress(result: dict, done: int, total: int):
    status = result.get("status")
    video_id = result.get("video_id")
    if status == "ok":
        print(
            f"[done] {done}/{total} video_id={video_id} frames={result.get('sampled_frames')} "
            f"latency={result.get('latency_seconds')}s",
            flush=True,
        )
    else:
        print(
            f"[failed] {done}/{total} video_id={video_id} error={result.get('error')}",
            flush=True,
        )


def print_heartbeat(completed: int, total: int, running: int, pending: int, elapsed_seconds: float):
    print(
        f"[progress] completed={completed}/{total} running={running} "
        f"pending={pending} elapsed_seconds={elapsed_seconds:.1f}",
        flush=True,
    )


def sync_result_to_metadata_db(store: MetadataStore, result: dict):
    if result.get("status") != "ok":
        return
    description = str(result.get("description") or result.get("caption") or "").strip()
    caption_text = str(result.get("caption") or description).strip()
    tags = result.get("tags") or []
    if not description or not tags:
        return
    payload = {
        "tags": tags,
        "description": description,
    }
    store.upsert_video_caption_metadata(
        video_id=result["video_id"],
        path=result.get("video_path"),
        duration=result.get("duration_seconds"),
        tags_json=json.dumps(tags, ensure_ascii=False),
        description=description,
        payload_json=json.dumps(payload, ensure_ascii=False),
        caption_model=result.get("model"),
        caption_updated_at=result.get("created_at"),
    )
    store.upsert_search_document(
        video_id=result["video_id"],
        path=result.get("video_path"),
        duration=result.get("duration_seconds"),
        tags_json=json.dumps(tags, ensure_ascii=False),
        description=description,
        caption_text=caption_text,
        sparse_tags=build_tag_term_text(tags),
        sparse_description=build_description_term_text(description),
        sparse_caption=build_caption_term_text(tags, description, caption_text),
        search_payload=json.dumps(payload, ensure_ascii=False),
        caption_model=result.get("model"),
        caption_updated_at=result.get("created_at"),
    )


def run_batch(config: CaptionConfig):
    if not config.video_dir.exists():
        raise RuntimeError(f"Video directory not found: {config.video_dir}")

    existing_rows = read_existing_results(config.output_jsonl)
    pending_videos = select_videos(config, existing_rows)
    print(
        f"[queue] video_dir={config.video_dir} total_existing={len(existing_rows)} "
        f"pending={len(pending_videos)} workers={config.workers}",
        flush=True,
    )
    if not pending_videos:
        print("[queue] nothing to do", flush=True)
        return
    pending_ids = {path.stem for path in pending_videos}
    retained_rows = [row for video_id, row in existing_rows.items() if video_id not in pending_ids]
    if config.output_jsonl.exists():
        rewrite_jsonl(config.output_jsonl, retained_rows)

    completed = 0
    success = 0
    failure = 0
    started_at = time.time()
    metadata_store = (
        MetadataStore(config.metadata_db_path)
        if config.sync_metadata_db and config.metadata_db_path is not None
        else None
    )
    try:
        with ThreadPoolExecutor(max_workers=config.workers) as executor:
            future_map = {
                executor.submit(process_video, config, video_path): video_path
                for video_path in pending_videos
            }
            remaining_futures = set(future_map)
            print_heartbeat(
                completed=0,
                total=len(pending_videos),
                running=min(config.workers, len(pending_videos)),
                pending=max(0, len(pending_videos) - min(config.workers, len(pending_videos))),
                elapsed_seconds=0.0,
            )
            while remaining_futures:
                done_futures, not_done_futures = wait(
                    remaining_futures,
                    timeout=config.progress_interval_seconds,
                    return_when=FIRST_COMPLETED,
                )
                if not done_futures:
                    running = min(len(not_done_futures), config.workers)
                    queued_pending = max(0, len(not_done_futures) - running)
                    print_heartbeat(
                        completed=completed,
                        total=len(pending_videos),
                        running=running,
                        pending=queued_pending,
                        elapsed_seconds=time.time() - started_at,
                    )
                    continue

                for future in done_futures:
                    result = future.result()
                    append_jsonl(config.output_jsonl, result)
                    if metadata_store is not None:
                        sync_result_to_metadata_db(metadata_store, result)
                    completed += 1
                    if result.get("status") == "ok":
                        success += 1
                    else:
                        failure += 1
                    print_progress(result, completed, len(pending_videos))
                remaining_futures = set(not_done_futures)
    finally:
        if metadata_store is not None:
            metadata_store.close()

    elapsed_seconds = round(time.time() - started_at, 3)
    print(
        f"[summary] success={success} failed={failure} elapsed_seconds={elapsed_seconds} "
        f"output={config.output_jsonl}",
        flush=True,
    )
    if config.output_json is not None:
        rows = read_existing_results(config.output_jsonl)
        aggregated = {
            video_id: {
                "caption": row.get("caption"),
                "description": row.get("description") or row.get("caption"),
                "tags": row.get("tags") or [],
                "structured_caption": row.get("structured_caption"),
                "status": row.get("status"),
                "video_path": row.get("video_path"),
                "error": row.get("error"),
                "sampled_frames": row.get("sampled_frames"),
                "duration_seconds": row.get("duration_seconds"),
                "model": row.get("model"),
                "created_at": row.get("created_at"),
            }
            for video_id, row in rows.items()
        }
        write_json(config.output_json, aggregated)
        print(f"[summary] output_json={config.output_json}", flush=True)


def main():
    args = parse_args()
    config = build_config(args)
    print(f"[run] start_time={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    config_payload = asdict(config)
    config_payload["api_key"] = "***redacted***"
    config_payload["opencv_log_level"] = os.environ.get("OPENCV_LOG_LEVEL")
    config_payload["opencv_ffmpeg_debug"] = os.environ.get("OPENCV_FFMPEG_DEBUG")
    config_payload["opencv_ffmpeg_loglevel"] = os.environ.get("OPENCV_FFMPEG_LOGLEVEL")
    print(f"[config] {json.dumps(config_payload, ensure_ascii=False, default=str)}", flush=True)
    run_batch(config)
    print(f"[run] end_time={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
