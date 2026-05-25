"""Preflight checks before loading CLIP / hybrid retrieval engines."""

from __future__ import annotations

from pathlib import Path

from .profile_paths import ProfileLayout

CLIP_FAISS_FILES = (
    "frame_index.faiss",
    "frame_index.meta.json",
    "segment_index.faiss",
    "segment_index.meta.json",
)


def _missing(path: Path, label: str) -> str | None:
    if path.exists():
        return None
    return f"{label} 不存在: {path}"


def check_model_path(model_path: Path) -> list[str]:
    issues: list[str] = []
    if not model_path.exists():
        issues.append(f"Chinese-CLIP 模型目录不存在: {model_path}")
        return issues
    has_weight = any(
        model_path.glob(pattern)
        for pattern in ("*.bin", "*.pt", "*.safetensors", "config.json")
    )
    if not has_weight and not any(model_path.iterdir()):
        issues.append(f"模型目录为空: {model_path}")
    return issues


def check_clip_artifacts(layout: ProfileLayout) -> list[str]:
    issues: list[str] = []
    faiss_dir = layout.clip_output_dir / "faiss"
    for name in CLIP_FAISS_FILES:
        if msg := _missing(faiss_dir / name, f"CLIP 索引 {name}"):
            issues.append(msg)
    if msg := _missing(layout.clip_metadata_db, "CLIP metadata.db"):
        issues.append(msg)
    return issues


def check_hybrid_artifacts(layout: ProfileLayout) -> list[str]:
    issues: list[str] = []
    index_dir = layout.hybrid_index_dir
    if not index_dir.is_dir():
        issues.append(f"Hybrid 索引目录不存在: {index_dir}")
        return issues
    manifest_files = ("embedder_manifest.json", "index_manifest.json")
    if not any((index_dir / name).is_file() for name in manifest_files):
        issues.append(
            f"Hybrid 索引目录缺少 manifest（{', '.join(manifest_files)}）: {index_dir}"
        )
    if msg := _missing(layout.hybrid_metadata_db, "Hybrid metadata.db"):
        issues.append(msg)
    return issues


def run_startup_checks(
    layout: ProfileLayout,
    *,
    model_path: Path,
    need_clip: bool = True,
    need_hybrid: bool = True,
    strict: bool = False,
) -> list[str]:
    """Return human-readable issues. If strict, raise RuntimeError when any issue exists."""
    issues: list[str] = []
    issues.extend(check_model_path(model_path))
    if need_clip:
        issues.extend(check_clip_artifacts(layout))
    if need_hybrid:
        issues.extend(check_hybrid_artifacts(layout))
    if strict and issues:
        raise RuntimeError("启动检查未通过:\n- " + "\n- ".join(issues))
    return issues
