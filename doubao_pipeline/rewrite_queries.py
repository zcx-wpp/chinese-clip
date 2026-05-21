from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from .config import PIPELINE_ROOT, WORKSPACE_ROOT


DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_API_KEY_ENV = "ARK_API_KEY"
DEFAULT_MODEL_ENV = "DOUBAO_MODEL"
DEFAULT_INPUT_PATH = WORKSPACE_ROOT / "data" / "captions_doubao_3000_queries.txt"
DEFAULT_OUTPUT_PATH = WORKSPACE_ROOT / "data" / "captions_doubao_3000_queries_rewritten.txt"
DEFAULT_SYSTEM_PROMPT = (
    "你是中文视频检索 query 改写助手。"
    "请把每条输入 query 改写成意思等价的另一种简体中文表达。"
    "可以适度扩写，也可以适度缩写，但绝不能改变原意，不能引入原文没有的新事实。"
    "必须保留关键主体、动作、场景、物体和关系。"
    "改写后的句子在表述上必须明显不同于原句，不能原样照抄，也不要只改动个别标点。"
    "输出必须是一个 JSON 对象，格式为 "
    '{"queries":[{"index":0,"rewrite":"..."},{"index":1,"rewrite":"..."}]}。'
    "数组长度必须与输入完全一致，顺序必须一致。"
    "每条 rewrite 只能是一行简体中文，不要编号，不要解释，不要输出 JSON 之外的任何内容。"
)

_THREAD_STATE = threading.local()


@dataclass(frozen=True)
class RewriteConfig:
    input_txt: Path
    output_txt: Path
    api_key: str
    model: str
    base_url: str
    workers: int
    batch_size: int
    max_retries: int
    temperature: float
    max_tokens: int
    timeout_seconds: float
    progress_interval_seconds: float
    overwrite: bool


def parse_args():
    parser = argparse.ArgumentParser(description="Rewrite retrieval queries with Doubao while preserving meaning.")
    parser.add_argument("--input-txt", default=str(DEFAULT_INPUT_PATH))
    parser.add_argument("--output-txt", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--model", help=f"Override model name. Defaults to {DEFAULT_MODEL_ENV} from .env.")
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max-tokens", type=int, default=1600)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--progress-interval-seconds", type=float, default=5.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def parse_simple_dotenv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        values[key] = value
    return values


def load_default_dotenv_files():
    for dotenv_path in (WORKSPACE_ROOT / ".env", PIPELINE_ROOT / ".env"):
        if not dotenv_path.exists():
            continue
        if load_dotenv is not None:
            load_dotenv(dotenv_path=dotenv_path, override=False)
            continue
        for key, value in parse_simple_dotenv(dotenv_path).items():
            os.environ.setdefault(key, value)


def build_config(args) -> RewriteConfig:
    load_default_dotenv_files()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key environment variable: {args.api_key_env}")
    model = args.model or os.environ.get(DEFAULT_MODEL_ENV)
    if not model:
        raise RuntimeError(f"Missing model. Pass --model or set {DEFAULT_MODEL_ENV} in .env.")
    return RewriteConfig(
        input_txt=Path(args.input_txt),
        output_txt=Path(args.output_txt),
        api_key=api_key,
        model=model,
        base_url=args.base_url,
        workers=max(1, args.workers),
        batch_size=max(1, args.batch_size),
        max_retries=max(1, args.max_retries),
        temperature=args.temperature,
        max_tokens=max(256, args.max_tokens),
        timeout_seconds=max(10.0, args.timeout_seconds),
        progress_interval_seconds=max(1.0, args.progress_interval_seconds),
        overwrite=args.overwrite,
    )


def normalize_query(text: str) -> str:
    return " ".join(str(text).replace("\r", " ").replace("\n", " ").split()).strip()


def surface_key(text: str) -> str:
    normalized = normalize_query(text)
    return re.sub(r"[\s，。、“”‘’！？：；,.!?:;()（）【】\[\]{}<>《》\"'`~·…-]+", "", normalized)


def read_queries(path: Path) -> list[str]:
    return [normalize_query(line) for line in path.read_text(encoding="utf-8").splitlines()]


def read_existing_output(path: Path, expected_count: int) -> list[str]:
    if not path.exists():
        return [""] * expected_count
    lines = [normalize_query(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if len(lines) < expected_count:
        lines.extend([""] * (expected_count - len(lines)))
    elif len(lines) > expected_count:
        lines = lines[:expected_count]
    return lines


def write_queries(path: Path, queries: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(queries)
    if queries:
        payload += "\n"
    path.write_text(payload, encoding="utf-8")


def get_client(config: RewriteConfig) -> OpenAI:
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


def extract_text_from_chat_response(response) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                if item.strip():
                    parts.append(item.strip())
                continue
            text = getattr(item, "text", None)
            if text:
                parts.append(str(text).strip())
                continue
            if isinstance(item, dict):
                maybe_text = item.get("text")
                if maybe_text:
                    parts.append(str(maybe_text).strip())
        return "\n".join(part for part in parts if part).strip()
    return ""


def build_messages(batch: list[tuple[int, str]]) -> list[dict]:
    payload = {
        "queries": [
            {"index": local_index, "text": text}
            for local_index, (_, text) in enumerate(batch)
        ]
    }
    user_prompt = (
        "请改写下面这些检索 query，并严格按要求返回 JSON。\n"
        f"{json.dumps(payload, ensure_ascii=False)}"
    )
    return [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def parse_rewrite_response(text: str, expected_count: int) -> list[str]:
    json_text = extract_first_json_object(text) or strip_code_fence(text)
    payload = json.loads(json_text)
    items = payload.get("queries") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        raise RuntimeError("Rewrite API did not return a JSON queries list.")

    rewrites = [""] * expected_count
    if items and all(isinstance(item, str) for item in items):
        if len(items) != expected_count:
            raise RuntimeError(
                f"Rewrite API returned {len(items)} items, expected {expected_count}."
            )
        for index, value in enumerate(items):
            rewrites[index] = normalize_query(value)
    else:
        for item in items:
            if not isinstance(item, dict):
                raise RuntimeError("Rewrite API returned a malformed queries item.")
            index = item.get("index")
            if not isinstance(index, int) or not (0 <= index < expected_count):
                raise RuntimeError(f"Rewrite API returned an invalid item index: {index!r}")
            rewritten = (
                item.get("rewrite")
                or item.get("text")
                or item.get("query")
                or item.get("output")
                or ""
            )
            rewrites[index] = normalize_query(rewritten)

    if any(not value for value in rewrites):
        raise RuntimeError("Rewrite API returned empty text for one or more queries.")
    return rewrites


def rewrite_batch(config: RewriteConfig, batch: list[tuple[int, str]]) -> list[tuple[int, str]]:
    client = get_client(config)
    last_error: Exception | None = None
    for attempt in range(1, config.max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=config.model,
                messages=build_messages(batch),
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
            response_text = extract_text_from_chat_response(response)
            rewrites = parse_rewrite_response(response_text, expected_count=len(batch))
            unchanged_indexes = [
                global_index
                for (global_index, original), rewritten in zip(batch, rewrites)
                if surface_key(original) == surface_key(rewritten)
            ]
            if unchanged_indexes:
                raise RuntimeError(
                    f"Rewrite API returned unchanged text for {len(unchanged_indexes)} queries."
                )
            return [
                (global_index, rewritten)
                for (global_index, _), rewritten in zip(batch, rewrites)
            ]
        except Exception as exc:
            last_error = exc
            if attempt < config.max_retries:
                time.sleep(min(5.0, attempt * 1.5))
    raise RuntimeError(f"Failed to rewrite batch after {config.max_retries} attempts: {last_error}")


def make_batches(indexes: list[int], queries: list[str], batch_size: int) -> list[list[tuple[int, str]]]:
    batches: list[list[tuple[int, str]]] = []
    for start in range(0, len(indexes), batch_size):
        batch_indexes = indexes[start : start + batch_size]
        batches.append([(index, queries[index]) for index in batch_indexes])
    return batches


def main():
    args = parse_args()
    config = build_config(args)
    queries = read_queries(config.input_txt)
    outputs = read_existing_output(config.output_txt, expected_count=len(queries))
    completed_initial = 0
    pending_indexes: list[int] = []
    for index, query in enumerate(queries):
        if not query:
            outputs[index] = ""
            completed_initial += 1
            continue
        if not config.overwrite and outputs[index]:
            completed_initial += 1
            continue
        pending_indexes.append(index)

    print(
        "[config] "
        + json.dumps(
            {
                "input_txt": str(config.input_txt),
                "output_txt": str(config.output_txt),
                "model": config.model,
                "base_url": config.base_url,
                "workers": config.workers,
                "batch_size": config.batch_size,
                "max_retries": config.max_retries,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "timeout_seconds": config.timeout_seconds,
                "overwrite": config.overwrite,
                "total_queries": len(queries),
                "pending_queries": len(pending_indexes),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    if not pending_indexes:
        write_queries(config.output_txt, outputs)
        print(f"[done] no pending queries output={config.output_txt}", flush=True)
        return

    batches = make_batches(pending_indexes, queries, config.batch_size)
    completed_queries = completed_initial
    completed_batches = 0
    total_batches = len(batches)
    started_at = time.time()
    write_queries(config.output_txt, outputs)
    print(
        f"[queue] batches={total_batches} pending_queries={len(pending_indexes)} workers={config.workers}",
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        future_to_batch = {
            executor.submit(rewrite_batch, config, batch): batch
            for batch in batches
        }
        while future_to_batch:
            done, _ = wait(
                list(future_to_batch.keys()),
                timeout=config.progress_interval_seconds,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                elapsed = time.time() - started_at
                print(
                    f"[progress] queries={completed_queries}/{len(queries)} "
                    f"batches={completed_batches}/{total_batches} elapsed_seconds={elapsed:.1f}",
                    flush=True,
                )
                continue
            for future in done:
                batch = future_to_batch.pop(future)
                results = future.result()
                for index, rewritten in results:
                    outputs[index] = rewritten
                completed_queries += len(batch)
                completed_batches += 1
                write_queries(config.output_txt, outputs)
                elapsed = time.time() - started_at
                print(
                    f"[done] queries={completed_queries}/{len(queries)} "
                    f"batches={completed_batches}/{total_batches} elapsed_seconds={elapsed:.1f}",
                    flush=True,
                )

    elapsed = time.time() - started_at
    write_queries(config.output_txt, outputs)
    print(
        f"[summary] total_queries={len(queries)} output={config.output_txt} elapsed_seconds={elapsed:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
