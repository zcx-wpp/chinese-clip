from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class MultimodalSignals:
    ocr_text: str = ""
    asr_text: str = ""

    def merged_text(self) -> str:
        parts = [part.strip() for part in [self.ocr_text, self.asr_text] if part and part.strip()]
        return " ".join(parts)


class NoOpOcrEngine:
    def extract(self, image_path: Path) -> str:
        return ""


class NoOpAsrEngine:
    def transcribe(self, video_path: Path) -> str:
        return ""


class PaddleOcrEngine:
    def __init__(self, lang: str = "ch"):
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "PaddleOCR is not installed. Install it with: pip install paddleocr"
            ) from exc
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang)

    def extract(self, image_path: Path) -> str:
        result = self.ocr.ocr(str(image_path), cls=True)
        texts = []
        for block in result or []:
            for line in block or []:
                if len(line) >= 2 and isinstance(line[1], (list, tuple)) and line[1]:
                    text = str(line[1][0]).strip()
                    if text:
                        texts.append(text)
        return " ".join(texts)


class WhisperAsrEngine:
    def __init__(self, model_name: str = "base"):
        try:
            import whisper
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "openai-whisper is not installed. Install it with: pip install openai-whisper"
            ) from exc
        self.model = whisper.load_model(model_name)

    def transcribe(self, video_path: Path) -> str:
        result = self.model.transcribe(str(video_path), fp16=False)
        return str(result.get("text", "")).strip()


def build_ocr_engine(enable_ocr: bool, lang: str = "ch"):
    if not enable_ocr:
        return NoOpOcrEngine()
    return PaddleOcrEngine(lang=lang)


def build_asr_engine(enable_asr: bool, model_name: str = "base"):
    if not enable_asr:
        return NoOpAsrEngine()
    return WhisperAsrEngine(model_name=model_name)
