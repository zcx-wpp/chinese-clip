from __future__ import annotations

import json
import re
from dataclasses import dataclass

DEFAULT_IMAGE_CAPTION_PROMPT = "你将看到一张图片。请基于画面内容生成适合图片检索的结构化描述。使用简体中文。"
STRUCTURED_OUTPUT_INSTRUCTIONS = (
    '请只输出 JSON：{"subject":"","color":"","action":"","style":"","description":""}。'
)


@dataclass
class StructuredImageCaption:
    subject: str = ""
    color: str = ""
    action: str = ""
    style: str = ""
    description: str = ""

    def is_valid(self) -> bool:
        return bool(self.description.strip() or self.subject.strip())

    def to_display_line(self) -> str:
        return (
            f"主体: [{self.subject or '-'}] | 颜色: [{self.color or '-'}] | "
            f"动作: [{self.action or '-'}] | 风格: [{self.style or '-'}]"
        )

    def to_embedding_text(self) -> str:
        parts = [self.to_display_line()]
        if self.description.strip():
            parts.append(f"描述: {self.description.strip()}")
        return " ".join(parts)

    def to_dict(self) -> dict[str, str]:
        return {
            "subject": self.subject,
            "color": self.color,
            "action": self.action,
            "style": self.style,
            "description": self.description,
        }


def build_structured_prompt(base: str | None = None) -> str:
    return f"{(base or DEFAULT_IMAGE_CAPTION_PROMPT).strip()}\n{STRUCTURED_OUTPUT_INSTRUCTIONS}"


def parse_structured_caption(text: str) -> StructuredImageCaption:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if m:
        raw = m.group(0)
        try:
            import json_repair
            payload = json_repair.loads(raw)
        except ImportError:
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = None
        except Exception:
            payload = None
        if isinstance(payload, dict):
            cap = StructuredImageCaption(
                subject=str(payload.get("subject") or payload.get("主体") or "").strip(),
                color=str(payload.get("color") or payload.get("颜色") or "").strip(),
                action=str(payload.get("action") or payload.get("动作") or "").strip(),
                style=str(payload.get("style") or payload.get("风格") or "").strip(),
                description=str(payload.get("description") or payload.get("描述") or "").strip(),
            )
            if cap.is_valid():
                return cap
    return StructuredImageCaption(description=cleaned or text.strip())
