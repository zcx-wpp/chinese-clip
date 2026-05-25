from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI
from PIL import Image

from .caption_schema import (
    StructuredImageCaption,
    build_structured_prompt,
    parse_structured_caption,
)
from .config import (
    DEFAULT_MLLM_BASE_URL,
    FALLBACK_MLLM_API_KEY_ENV,
    FALLBACK_MLLM_MODEL_ENV,
    LEGACY_MLLM_MODEL_ENV,
    PICTURE_MLLM_API_KEY_ENV,
    PICTURE_MLLM_MODEL_ENV,
)
from .env_loader import env_first, load_default_dotenv_files


def encode_jpg_data_url(image: Image.Image, *, max_side: int = 1280) -> str:
    img = image.convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / max(w, h)) if max(w, h) > max_side else 1.0
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode('ascii')}"


@dataclass
class MllmCaptionConfig:
    api_key: str
    model: str
    base_url: str = DEFAULT_MLLM_BASE_URL
    api_mode: str = "chat"
    temperature: float = 0.2
    max_tokens: int = 512
    timeout_seconds: float = 120.0


def resolve_mllm_config(**kw) -> MllmCaptionConfig:
    load_default_dotenv_files()
    key = kw.get("api_key") or env_first(PICTURE_MLLM_API_KEY_ENV, FALLBACK_MLLM_API_KEY_ENV)
    model = kw.get("model") or env_first(
        PICTURE_MLLM_MODEL_ENV, FALLBACK_MLLM_MODEL_ENV, LEGACY_MLLM_MODEL_ENV
    )
    if not key or not model:
        raise RuntimeError("Missing MLLM API key/model in .env")
    return MllmCaptionConfig(
        api_key=key, model=model, base_url=kw.get("base_url") or DEFAULT_MLLM_BASE_URL
    )


class ImageMllmCaptioner:
    def __init__(self, config: MllmCaptionConfig):
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key, base_url=config.base_url, timeout=config.timeout_seconds
        )

    def caption_image_path(self, path: Path) -> StructuredImageCaption:
        with Image.open(path) as img:
            return self.caption_pil_image(img)

    def caption_pil_image(self, image: Image.Image) -> StructuredImageCaption:
        prompt = build_structured_prompt()
        data_url = encode_jpg_data_url(image)
        raw = self._call_api(prompt, data_url)
        cap = parse_structured_caption(raw)
        if not cap.is_valid():
            raise RuntimeError(f"empty caption: {raw[:200]!r}")
        return cap

    def _call_api(self, prompt: str, data_url: str) -> str:
        try:
            from tenacity import retry, stop_after_attempt, wait_exponential

            @retry(reraise=True, stop=stop_after_attempt(5), wait=wait_exponential(min=2, max=60))
            def _go():
                return self._call_api_once(prompt, data_url)

            return _go()
        except ImportError:
            return self._call_api_once(prompt, data_url)

    def _call_api_once(self, prompt: str, data_url: str) -> str:
        if self.config.api_mode == "responses":
            response = self.client.responses.create(
                model=self.config.model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": data_url},
                        ],
                    }
                ],
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_tokens,
            )
            text = getattr(response, "output_text", None)
            if text:
                return str(text).strip()
            raise RuntimeError("empty responses API output")

        r = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        c = r.choices[0].message.content
        return c.strip() if isinstance(c, str) else str(c)
