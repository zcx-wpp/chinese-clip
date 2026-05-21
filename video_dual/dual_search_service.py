from __future__ import annotations

import argparse
import mimetypes
import threading
import time
from pathlib import Path
from urllib.parse import quote

from pydantic import BaseModel, Field

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CLIP_PROFILE = "apr_media1_project"
DEFAULT_DOUBAO_PROFILE = "apr_media1"
DEFAULT_MODEL_PATH = str(WORKSPACE_ROOT / "project" / "models")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1, le=50)


def _format_segments(segments: list[dict]) -> str:
    if not segments:
        return ""
    parts = []
    for seg in segments[:3]:
        start = seg.get("start", 0)
        end = seg.get("end", start)
        score = seg.get("score")
        if score is not None:
            parts.append(f"{start:.1f}-{end:.1f}s ({score:.3f})")
        else:
            parts.append(f"{start:.1f}-{end:.1f}s")
    return " · ".join(parts)


def _normalize_clip_hit(item: dict) -> dict:
    segments = list(item.get("segments") or [])
    seg_line = _format_segments(segments)
    return {
        "video_id": str(item.get("video_id") or ""),
        "score": float(item.get("score") or 0.0),
        "description": seg_line,
        "display_line": seg_line or "Chinese-CLIP 帧级检索",
        "tags": [],
        "segments": segments,
        "video_path": str(item.get("video_path") or ""),
        "sparse_rank": None,
        "dense_rank": None,
    }


def _normalize_doubao_hit(item: dict) -> dict:
    tags = item.get("tags") or []
    if not isinstance(tags, list):
        tags = [str(tags)]
    description = str(item.get("description") or item.get("caption") or "").strip()
    return {
        "video_id": str(item.get("video_id") or ""),
        "score": float(item.get("score") or 0.0),
        "description": description,
        "display_line": description,
        "tags": [str(t) for t in tags if str(t).strip()],
        "segments": list(item.get("segments") or []),
        "video_path": str(item.get("video_path") or ""),
        "sparse_rank": item.get("sparse_rank"),
        "dense_rank": item.get("dense_rank"),
    }


class DoubaoSearchBridge:
    """Lazy-loaded doubao_pipeline hybrid engine."""

    def __init__(
        self,
        *,
        profile: str | None,
        metadata_db: Path | None,
        index_dir: Path | None,
        embedding_device: str,
        embedding_batch_size: int,
        embedding_local_files_only: bool,
        sparse_top_k: int,
        dense_top_k: int,
        rrf_k: int,
    ):
        from doubao_pipeline.hybrid_retrieval import HybridSearchConfig, build_search_engine
        from doubao_pipeline.profile_paths import resolve_search_sources

        self._build_search_engine = build_search_engine
        self._resolve_search_sources = resolve_search_sources
        self._HybridSearchConfig = HybridSearchConfig
        self.profile = profile
        self.metadata_db = metadata_db
        self.index_dir = index_dir
        self.embedding_device = embedding_device
        self.embedding_batch_size = embedding_batch_size
        self.embedding_local_files_only = embedding_local_files_only
        self.search_config = HybridSearchConfig(
            sparse_top_k=sparse_top_k,
            dense_top_k=dense_top_k,
            rrf_k=rrf_k,
        )
        self._engine = None
        self._starting = False
        self._error: str | None = None
        self._lock = threading.Lock()

    def _ensure_started(self) -> None:
        with self._lock:
            if self._engine is not None or self._starting:
                return
            self._starting = True
            self._error = None

        engine = None
        error = None
        try:
            sources = self._resolve_search_sources(
                profile=self.profile,
                metadata_db=self.metadata_db,
                index_dir=self.index_dir,
            )
            engine = self._build_search_engine(
                sources=sources,
                search_config=self.search_config,
                embedding_device=self.embedding_device,
                embedding_batch_size=self.embedding_batch_size,
                embedding_local_files_only=self.embedding_local_files_only,
            )
            engine.warmup()
        except Exception as exc:
            error = str(exc)

        with self._lock:
            self._engine = engine
            self._error = error
            self._starting = False

    def start_background(self) -> None:
        threading.Thread(target=self._ensure_started, daemon=True).start()

    def status(self) -> dict:
        with self._lock:
            return {
                "ready": self._engine is not None,
                "starting": self._starting,
                "error": self._error,
                "profile": self.profile,
            }

    def require_engine(self):
        self._ensure_started()
        with self._lock:
            if self._engine is not None:
                return self._engine
            if self._starting:
                raise RuntimeError("Doubao 检索引擎加载中，请稍后重试。")
            raise RuntimeError(self._error or "Doubao 检索引擎不可用")

    def index_count(self) -> int:
        try:
            engine = self.require_engine()
            return len(getattr(engine, "documents_by_id", {}) or {})
        except RuntimeError:
            return 0

    def search(self, query: str, *, top_k: int) -> tuple[list[dict], float]:
        from doubao_pipeline.portable_paths import resolve_portable_path

        engine = self.require_engine()
        started = time.perf_counter()
        raw = engine.search(query, top_k=top_k)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        results = []
        for item in raw:
            row = _normalize_doubao_hit(item)
            raw_path = row.get("video_path") or ""
            if raw_path:
                row["video_path"] = str(resolve_portable_path(raw_path))
            results.append(row)
        return results, elapsed_ms

    def resolve_video_path(self, video_id: str) -> Path:
        from doubao_pipeline.portable_paths import resolve_portable_path

        engine = self.require_engine()
        row = engine.documents_by_id.get(video_id)
        if row is None:
            raise FileNotFoundError(f"video_id not found: {video_id}")
        raw_path = str(row.get("path") or "").strip()
        if not raw_path:
            raise FileNotFoundError(f"video path missing for {video_id}")
        path = resolve_portable_path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"video file missing: {path}")
        return path

    def close(self) -> None:
        with self._lock:
            engine = self._engine
            self._engine = None
        if engine is not None:
            engine.close()


class DualVideoSearchService:
    def __init__(
        self,
        *,
        clip_profile: str,
        clip_metadata_db: Path | None,
        clip_output_dir: Path | None,
        doubao_profile: str | None,
        doubao_metadata_db: Path | None,
        doubao_index_dir: Path | None,
        model_path: str,
        device: str = "cuda",
        doubao_device: str = "cuda",
        doubao_batch_size: int = 16,
        doubao_local_files_only: bool = False,
        sparse_top_k: int = 100,
        dense_top_k: int = 100,
        rrf_k: int = 60,
    ):
        from project.src.video_processing.api import build_retriever
        from project.src.video_processing.portable_paths import resolve_portable_path
        from project.src.video_processing.profile_paths import (
            default_metadata_db_path,
            default_output_dir,
        )

        self._resolve_portable_path = resolve_portable_path

        clip_out = clip_output_dir or default_output_dir(clip_profile)
        clip_db = clip_metadata_db or default_metadata_db_path(clip_profile)
        self.clip_profile = clip_profile
        self.clip = build_retriever(
            output_dir=clip_out,
            metadata_db_path=clip_db,
            model_path=model_path,
            device=device,
        )

        self.doubao = DoubaoSearchBridge(
            profile=doubao_profile,
            metadata_db=doubao_metadata_db,
            index_dir=doubao_index_dir,
            embedding_device=doubao_device,
            embedding_batch_size=doubao_batch_size,
            embedding_local_files_only=doubao_local_files_only,
            sparse_top_k=sparse_top_k,
            dense_top_k=dense_top_k,
            rrf_k=rrf_k,
        )
        self.doubao.start_background()

    def _attach_media(self, results: list[dict], *, mode: str) -> list[dict]:
        resolve = self.resolve_clip_path if mode == "clip" else self.resolve_doubao_path
        enriched = []
        for item in results:
            row = dict(item)
            vid = str(row.get("video_id") or "")
            path = None
            if row.get("video_path"):
                try:
                    candidate = Path(str(row["video_path"]))
                    path = candidate if candidate.is_file() else self._resolve_portable_path(str(row["video_path"]))
                except Exception:
                    path = None
            if path is None or not Path(path).is_file():
                path = resolve(vid)
            row["video_available"] = path is not None and Path(path).is_file()
            if row["video_available"]:
                row["resolved_path"] = str(path)
                row["video_url"] = f"/media/{mode}/{quote(vid, safe='')}"
            enriched.append(row)
        return enriched

    def search(self, query: str, *, top_k: int = 10) -> dict:
        started = time.perf_counter()
        clip_raw, clip_ms = self._search_clip(query, top_k=top_k)
        try:
            doubao_raw, doubao_ms = self.doubao.search(query, top_k=top_k)
        except RuntimeError as exc:
            doubao_raw, doubao_ms = [], 0.0
            doubao_error = str(exc)
        else:
            doubao_error = None

        clip_results = self._attach_media(clip_raw, mode="clip")
        doubao_results = self._attach_media(doubao_raw, mode="doubao")

        return {
            "query": query,
            "top_k": top_k,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 1),
            "clip": {
                "mode": "project_chinese_clip",
                "profile": self.clip_profile,
                "index_count": len(self.clip.index.item_ids),
                "elapsed_ms": round(clip_ms, 1),
                "results": clip_results,
            },
            "doubao": {
                "mode": "doubao_hybrid_sparse_dense",
                "index_count": self.doubao.index_count(),
                "elapsed_ms": round(doubao_ms, 1),
                "ready": self.doubao.status()["ready"],
                "error": doubao_error or self.doubao.status().get("error"),
                "results": doubao_results,
            },
        }

    def _search_clip(self, query: str, *, top_k: int) -> tuple[list[dict], float]:
        started = time.perf_counter()
        raw = self.clip.search(query, top_k=top_k)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return [_normalize_clip_hit(item) for item in raw], elapsed_ms

    def resolve_clip_path(self, video_id: str) -> Path | None:
        records = self.clip.metadata_store.get_video_records([video_id])
        if not records:
            return None
        raw_path = str(records[0].get("path") or "").strip()
        if not raw_path:
            return None
        try:
            path = self._resolve_portable_path(raw_path)
            return path if path.is_file() else None
        except Exception:
            return None

    def resolve_doubao_path(self, video_id: str) -> Path | None:
        try:
            return self.doubao.resolve_video_path(video_id)
        except (FileNotFoundError, RuntimeError):
            return None

    def health(self) -> dict:
        doubao_status = self.doubao.status()
        return {
            "clip_index_frames": len(self.clip.index.item_ids),
            "clip_profile": self.clip_profile,
            "doubao_index_videos": self.doubao.index_count(),
            "doubao_ready": doubao_status["ready"],
            "doubao_starting": doubao_status["starting"],
            "doubao_error": doubao_status.get("error"),
        }

    def close(self) -> None:
        self.clip.metadata_store.close()
        self.doubao.close()


def create_app(service: DualVideoSearchService):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse

    app = FastAPI(title="Video Dual Search", version="1.0.0")

    @app.get("/", response_class=HTMLResponse)
    def home():
        return INDEX_HTML

    @app.post("/search")
    def search(req: SearchRequest):
        return service.search(req.query, top_k=req.top_k)

    @app.get("/health")
    def health():
        return service.health()

    @app.get("/media/clip/{video_id:path}")
    def media_clip(video_id: str):
        from urllib.parse import unquote

        video_id = unquote(video_id).strip()
        path = service.resolve_clip_path(video_id)
        if path is None:
            raise HTTPException(404, detail=f"video not found: {video_id}")
        media_type = mimetypes.guess_type(path.name)[0] or "video/mp4"
        return FileResponse(path, media_type=media_type)

    @app.get("/media/doubao/{video_id:path}")
    def media_doubao(video_id: str):
        from urllib.parse import unquote

        video_id = unquote(video_id).strip()
        path = service.resolve_doubao_path(video_id)
        if path is None:
            raise HTTPException(404, detail=f"video not found: {video_id}")
        media_type = mimetypes.guess_type(path.name)[0] or "video/mp4"
        return FileResponse(path, media_type=media_type)

    return app


INDEX_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>双方案文搜视频对比</title>
  <style>
    * { box-sizing: border-box; }
    body { font-family: system-ui, "PingFang SC", sans-serif; margin: 0; padding: 20px; background: #0f1218; color: #e8ecf0; }
    h1 { margin: 0 0 8px; font-size: 1.4rem; }
    .sub { color: #8a9bb0; margin-bottom: 12px; font-size: 14px; }
    #engineStatus { font-size: 13px; color: #9ab; margin-bottom: 14px; min-height: 1.2em; }
    .bar { display: flex; gap: 10px; margin-bottom: 16px; flex-wrap: wrap; }
    input[type=text] { flex: 1; min-width: 240px; padding: 12px 14px; border-radius: 8px; border: 1px solid #334; background: #1a2230; color: #fff; font-size: 16px; }
    button { padding: 12px 22px; border: 0; border-radius: 8px; background: #57c084; color: #08110c; font-weight: 600; cursor: pointer; }
    button:disabled { opacity: 0.5; }
    #status { min-height: 1.4em; color: #8ab; margin-bottom: 12px; }
    .columns { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; align-items: start; }
    @media (max-width: 1000px) { .columns { grid-template-columns: 1fr; } }
    .panel { background: #161d28; border: 1px solid #2a3544; border-radius: 12px; padding: 14px; }
    .panel h2 { margin: 0 0 4px; font-size: 1.05rem; }
    .meta { font-size: 12px; color: #7a8fa8; margin-bottom: 12px; }
    .list { display: flex; flex-direction: column; gap: 12px; }
    .card { background: #1e2736; border-radius: 8px; overflow: hidden; border: 1px solid #2d3a4d; display: grid; grid-template-columns: 200px 1fr; gap: 0; }
    @media (max-width: 700px) { .card { grid-template-columns: 1fr; } }
    .card video { width: 100%; height: 112px; object-fit: cover; background: #000; display: block; }
    .card .thumb.missing { height: 112px; display: flex; align-items: center; justify-content: center; color: #f88; font-size: 11px; padding: 8px; text-align: center; }
    .card .info { padding: 10px; font-size: 12px; }
    .score { color: #6dd4a8; font-weight: 600; }
    .id { color: #9ab; word-break: break-all; margin: 4px 0; }
    .desc { color: #bcd; line-height: 1.4; max-height: 3.6em; overflow: hidden; }
    .tags { margin-top: 6px; display: flex; flex-wrap: wrap; gap: 4px; }
    .tag { background: #2a3544; padding: 2px 6px; border-radius: 4px; font-size: 11px; color: #9ab; }
    .rank-extra { color: #7a8fa8; font-size: 11px; margin-top: 4px; }
    .empty { color: #667; padding: 24px; text-align: center; }
    .warn { color: #e8a87c; }
  </style>
</head>
<body>
  <h1>双方案文搜视频对比</h1>
  <p class="sub">左：project · Chinese-CLIP 帧/片段检索　右：doubao_pipeline · 字幕稀疏 + BGE 稠密混合检索　各 Top 10</p>
  <div id="engineStatus">引擎状态加载中…</div>
  <div class="bar">
    <input type="text" id="query" placeholder="输入检索词，例如：一个人在厨房做饭" spellcheck="false" />
    <button type="button" id="go">检索</button>
  </div>
  <div id="status"></div>
  <div class="columns">
    <section class="panel">
      <h2>方案 A · project CLIP</h2>
      <div class="meta" id="clipMeta">—</div>
      <div class="list" id="clipList"></div>
    </section>
    <section class="panel">
      <h2>方案 B · Doubao 混合检索</h2>
      <div class="meta" id="doubaoMeta">—</div>
      <div class="list" id="doubaoList"></div>
    </section>
  </div>
  <script>
    const statusEl = document.getElementById("status");
    const engineStatusEl = document.getElementById("engineStatus");
    const queryEl = document.getElementById("query");
    const goBtn = document.getElementById("go");

    function escapeHtml(s) {
      return String(s)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;");
    }

    function cardHtml(item, mode) {
      const id = escapeHtml(item.video_id || "");
      const score = Number(item.score || 0).toFixed(4);
      const desc = escapeHtml(item.display_line || item.description || "");
      const tags = (item.tags || []).map((t) => `<span class="tag">${escapeHtml(t)}</span>`).join("");
      const extra = mode === "doubao" && (item.sparse_rank || item.dense_rank)
        ? `<div class="rank-extra">稀疏 #${item.sparse_rank || "-"} · 稠密 #${item.dense_rank || "-"}</div>`
        : "";
      const segs = (item.segments || []).length
        ? `<div class="rank-extra">片段 ${(item.segments || []).map(s => s.start + "-" + s.end + "s").join(", ")}</div>`
        : "";
      const thumb = item.video_available === false
        ? `<div class="thumb missing">视频缺失<br>${id}</div>`
        : `<video src="/media/${mode}/${encodeURIComponent(item.video_id || "")}" controls preload="metadata" playsinline></video>`;
      return `<article class="card">
        ${thumb}
        <div class="info">
          <div class="score">${score}</div>
          <div class="id">${id}</div>
          ${desc ? `<div class="desc">${desc}</div>` : ""}
          ${tags ? `<div class="tags">${tags}</div>` : ""}
          ${extra}${segs}
        </div>
      </article>`;
    }

    function renderList(listId, metaId, block, mode) {
      const list = document.getElementById(listId);
      const meta = document.getElementById(metaId);
      const items = (block && block.results) || [];
      let metaText = items.length
        ? `索引 ${block.index_count} · 用时 ${block.elapsed_ms} ms`
        : `无结果（索引 ${block ? block.index_count : 0}）`;
      if (mode === "doubao" && block && !block.ready) {
        metaText += block.error ? ` · <span class="warn">${escapeHtml(block.error)}</span>` : " · 引擎加载中";
      }
      meta.innerHTML = metaText;
      list.innerHTML = items.length
        ? items.map((it) => cardHtml(it, mode)).join("")
        : '<div class="empty">无命中</div>';
    }

    async function refreshHealth() {
      try {
        const res = await fetch("/health");
        const h = await res.json();
        const doubao = h.doubao_ready ? "Doubao 就绪" : (h.doubao_starting ? "Doubao 加载中…" : "Doubao 未就绪");
        engineStatusEl.textContent =
          `CLIP 帧索引 ${h.clip_index_frames}（${h.clip_profile}） · Doubao 视频 ${h.doubao_index_videos} · ${doubao}` +
          (h.doubao_error ? ` · ${h.doubao_error}` : "");
      } catch (e) {
        engineStatusEl.textContent = "状态获取失败";
      }
    }

    async function search() {
      const q = queryEl.value.trim();
      if (!q) return;
      goBtn.disabled = true;
      statusEl.textContent = "检索中…";
      try {
        const res = await fetch("/search", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: q, top_k: 10 }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || res.status);
        statusEl.textContent = `「${data.query}」总用时 ${data.elapsed_ms} ms`;
        renderList("clipList", "clipMeta", data.clip, "clip");
        renderList("doubaoList", "doubaoMeta", data.doubao, "doubao");
      } catch (e) {
        statusEl.textContent = "错误: " + e.message;
      } finally {
        goBtn.disabled = false;
      }
    }

    goBtn.onclick = search;
    queryEl.addEventListener("keydown", (e) => { if (e.key === "Enter") search(); });
    refreshHealth();
    setInterval(refreshHealth, 5000);
  </script>
</body>
</html>"""


def parse_args():
    p = argparse.ArgumentParser(description="Dual video text search (project CLIP + doubao hybrid).")
    p.add_argument("--clip-profile", default=DEFAULT_CLIP_PROFILE)
    p.add_argument("--clip-metadata-db", type=Path, default=None)
    p.add_argument("--clip-output-dir", type=Path, default=None)
    p.add_argument("--doubao-profile", default=DEFAULT_DOUBAO_PROFILE)
    p.add_argument("--doubao-metadata-db", type=Path, default=None)
    p.add_argument("--doubao-index-dir", type=Path, default=None)
    p.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    p.add_argument("--device", default="cuda")
    p.add_argument("--doubao-device", default="cuda")
    p.add_argument("--doubao-batch-size", type=int, default=16)
    p.add_argument("--doubao-local-files-only", action="store_true")
    p.add_argument("--sparse-top-k", type=int, default=100)
    p.add_argument("--dense-top-k", type=int, default=100)
    p.add_argument("--rrf-k", type=int, default=60)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8023)
    return p.parse_args()


def main():
    import uvicorn

    args = parse_args()
    from project.src.video_processing.profile_paths import (
        default_metadata_db_path as clip_default_db,
        default_output_dir as clip_default_out,
    )

    service = DualVideoSearchService(
        clip_profile=args.clip_profile,
        clip_metadata_db=args.clip_metadata_db or clip_default_db(args.clip_profile),
        clip_output_dir=args.clip_output_dir or clip_default_out(args.clip_profile),
        doubao_profile=args.doubao_profile,
        doubao_metadata_db=args.doubao_metadata_db,
        doubao_index_dir=args.doubao_index_dir,
        model_path=args.model_path,
        device=args.device,
        doubao_device=args.doubao_device,
        doubao_batch_size=args.doubao_batch_size,
        doubao_local_files_only=args.doubao_local_files_only,
        sparse_top_k=args.sparse_top_k,
        dense_top_k=args.dense_top_k,
        rrf_k=args.rrf_k,
    )
    app = create_app(service)
    print(f"[video_dual_search] http://{args.host}:{args.port}/", flush=True)
    print(
        f"[video_dual_search] clip={args.clip_profile} doubao={args.doubao_profile}",
        flush=True,
    )
    try:
        uvicorn.run(app, host=args.host, port=args.port)
    finally:
        service.close()


if __name__ == "__main__":
    main()
