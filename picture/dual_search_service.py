from __future__ import annotations

import argparse
import time
from pathlib import Path

from pydantic import BaseModel, Field

from .config import DEFAULT_MODEL_PATH, WORKSPACE_ROOT
from .profile_paths import default_output_dir, resolve_path
from .retrieval import build_retriever
from .text_retrieval import build_text_retriever


def default_image_search_roots(primary: Path) -> list[Path]:
    """Primary UI image-dir plus common MUGE folders for incremental ingest."""
    roots: list[Path] = [primary.resolve()]
    for rel in (
        "data/muge/train_extracted",
        "data/muge/dataset_1k/images",
        "data/muge/dataset_1k_add/images",
    ):
        candidate = (WORKSPACE_ROOT / rel).resolve()
        if candidate.is_dir() and candidate not in roots:
            roots.append(candidate)
    return roots


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1, le=50)


class DualSearchService:
    def __init__(
        self,
        *,
        clip_profile: str,
        bge_profile: str,
        image_dir: Path,
        model_path: str,
        device: str = "cuda",
        bge_device: str = "cuda",
    ):
        self.search_roots = default_image_search_roots(image_dir)
        self.image_dir = self.search_roots[0]
        self.clip = build_retriever(
            profile=clip_profile,
            model_path=model_path,
            device=device,
            search_roots=self.search_roots,
        )
        self.bge = build_text_retriever(
            profile=bge_profile,
            bge_device=bge_device,
            search_roots=self.search_roots,
        )

    def _attach_media_flags(self, results: list[dict], *, mode: str) -> list[dict]:
        enriched = []
        for item in results:
            iid = str(item.get("image_id") or "")
            if mode == "clip":
                path = self.resolve_clip_path(iid)
            else:
                path = self.resolve_bge_path(iid)
            row = dict(item)
            row["image_available"] = path is not None
            if path is not None:
                row["resolved_path"] = str(path)
            enriched.append(row)
        return enriched

    def search(self, query: str, *, top_k: int = 10) -> dict:
        started = time.perf_counter()
        clip_results, clip_ms = self.clip.search_text(query, top_k=top_k)
        bge_results, bge_ms = self.bge.search_text(query, top_k=top_k)
        total_ms = (time.perf_counter() - started) * 1000.0
        clip_results = self._attach_media_flags(clip_results, mode="clip")
        bge_results = self._attach_media_flags(bge_results, mode="bge")
        return {
            "query": query,
            "top_k": top_k,
            "elapsed_ms": round(total_ms, 1),
            "image_roots": [str(p) for p in self.search_roots],
            "clip": {
                "mode": "chinese_clip",
                "index_count": len(self.clip.index.item_ids),
                "elapsed_ms": round(clip_ms, 1),
                "results": clip_results,
            },
            "bge": {
                "mode": "mllm_bge",
                "index_count": self.bge.index_count(),
                "elapsed_ms": round(bge_ms, 1),
                "results": bge_results,
            },
        }

    def resolve_clip_path(self, image_id: str) -> Path | None:
        return self.clip.resolve_path(image_id)

    def resolve_bge_path(self, image_id: str) -> Path | None:
        return self.bge.resolve_path(image_id)

    def close(self) -> None:
        self.clip.close()
        self.bge.close()


def create_app(service: DualSearchService):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse

    app = FastAPI(title="Picture Dual Search", version="1.0.0")

    @app.get("/", response_class=HTMLResponse)
    def home():
        return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>双方案文搜图对比</title>
  <style>
    * { box-sizing: border-box; }
    body { font-family: system-ui, sans-serif; margin: 0; padding: 20px; background: #0f1218; color: #e8ecf0; }
    h1 { margin: 0 0 8px; font-size: 1.4rem; }
    .sub { color: #8a9bb0; margin-bottom: 16px; font-size: 14px; }
    .bar { display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }
    input[type=text] { flex: 1; min-width: 240px; padding: 12px 14px; border-radius: 8px; border: 1px solid #334; background: #1a2230; color: #fff; font-size: 16px; }
    button { padding: 12px 22px; border: 0; border-radius: 8px; background: #3d9a6f; color: #fff; font-weight: 600; cursor: pointer; }
    button:disabled { opacity: 0.5; }
    #status { min-height: 1.4em; color: #8ab; margin-bottom: 12px; }
    .columns { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; align-items: start; }
    @media (max-width: 900px) { .columns { grid-template-columns: 1fr; } }
    .panel { background: #161d28; border: 1px solid #2a3544; border-radius: 12px; padding: 14px; }
    .panel h2 { margin: 0 0 4px; font-size: 1.05rem; }
    .meta { font-size: 12px; color: #7a8fa8; margin-bottom: 12px; }
    .grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }
    @media (min-width: 1200px) { .grid { grid-template-columns: repeat(3, 1fr); } }
    .card { background: #1e2736; border-radius: 8px; overflow: hidden; border: 1px solid #2d3a4d; }
    .card .thumb { width: 100%; height: 120px; background: #0a0e14; display: flex; align-items: center; justify-content: center; overflow: hidden; }
    .card img { width: 100%; height: 120px; object-fit: cover; background: #000; display: block; }
    .card .thumb.missing { color: #f66; font-size: 11px; padding: 8px; text-align: center; }
    .card .info { padding: 8px; font-size: 12px; }
    .score { color: #6dd4a8; font-weight: 600; }
    .id { color: #9ab; word-break: break-all; }
    .tags { color: #bcd; margin-top: 4px; line-height: 1.35; max-height: 2.7em; overflow: hidden; }
    .empty { color: #667; padding: 24px; text-align: center; }
  </style>
</head>
<body>
  <h1>双方案文搜图对比</h1>
  <p class="sub">左：Chinese-CLIP（picture_image）　右：MLLM + BGE（picture_caption_bge）　各 Top 10</p>
  <div class="bar">
    <input type="text" id="query" placeholder="输入检索词，例如：橘猫 沙发" data-gramm="false" spellcheck="false" />
    <button type="button" id="go">检索</button>
  </div>
  <div id="status"></div>
  <div class="columns">
    <section class="panel">
      <h2>方案 A · Chinese-CLIP</h2>
      <div class="meta" id="clipMeta">—</div>
      <div class="grid" id="clipGrid"></div>
    </section>
    <section class="panel">
      <h2>方案 B · MLLM + BGE</h2>
      <div class="meta" id="bgeMeta">—</div>
      <div class="grid" id="bgeGrid"></div>
    </section>
  </div>
  <script>
    const statusEl = document.getElementById("status");
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
      const id = escapeHtml(item.image_id || "");
      const imgSrc = `/media/${mode}/${encodeURIComponent(item.image_id || "")}`;
      const tags = escapeHtml(item.display_line || item.description || "");
      const thumb = item.image_available === false
        ? `<div class="thumb missing">图片缺失<br>${id}</div>`
        : `<div class="thumb"><img src="${imgSrc}" alt="" loading="lazy"
             onerror="this.replaceWith(Object.assign(document.createElement('div'),{className:'thumb missing',innerHTML:'加载失败<br>${id}'}))" /></div>`;
      return `<div class="card">
        ${thumb}
        <div class="info">
          <div class="score">${Number(item.score).toFixed(4)}</div>
          <div class="id">${id}</div>
          ${tags ? `<div class="tags">${tags}</div>` : ""}
        </div>
      </div>`;
    }

    function renderColumn(gridId, metaId, block, mode) {
      const grid = document.getElementById(gridId);
      const meta = document.getElementById(metaId);
      const items = (block && block.results) || [];
      meta.textContent = items.length
        ? `索引 ${block.index_count} 条 · 用时 ${block.elapsed_ms} ms`
        : `无结果（索引 ${block ? block.index_count : 0} 条）`;
      if (!items.length) {
        grid.innerHTML = '<div class="empty">无命中</div>';
        return;
      }
      grid.innerHTML = items.map((it) => cardHtml(it, mode)).join("");
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
        renderColumn("clipGrid", "clipMeta", data.clip, "clip");
        renderColumn("bgeGrid", "bgeMeta", data.bge, "bge");
      } catch (e) {
        statusEl.textContent = "错误: " + e.message;
      } finally {
        goBtn.disabled = false;
      }
    }
    goBtn.onclick = search;
    queryEl.addEventListener("keydown", (e) => { if (e.key === "Enter") search(); });
  </script>
</body>
</html>"""

    @app.post("/search")
    def search(req: SearchRequest):
        return service.search(req.query, top_k=req.top_k)

    @app.get("/media/clip/{image_id:path}")
    def media_clip(image_id: str):
        from urllib.parse import unquote

        image_id = unquote(image_id).strip()
        path = service.resolve_clip_path(image_id)
        if path is None:
            raise HTTPException(404, detail=f"image not found: {image_id}")
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/media/bge/{image_id:path}")
    def media_bge(image_id: str):
        from urllib.parse import unquote

        image_id = unquote(image_id).strip()
        path = service.resolve_bge_path(image_id)
        if path is None:
            raise HTTPException(404, detail=f"image not found: {image_id}")
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/health")
    def health():
        return {
            "clip_index": len(service.clip.index.item_ids),
            "bge_index": service.bge.index_count(),
        }

    return app


def parse_args():
    p = argparse.ArgumentParser(description="Dual picture text search (CLIP + BGE side by side).")
    p.add_argument("--clip-profile", default="muge_1k_clip")
    p.add_argument("--bge-profile", default="muge_1k_bge")
    p.add_argument(
        "--image-dir",
        required=True,
        help="Primary image directory; also auto-tries train_extracted, dataset_1k_add.",
    )
    p.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    p.add_argument("--device", default="cuda")
    p.add_argument("--bge-device", default="cuda")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8022)
    return p.parse_args()


def main():
    import uvicorn

    args = parse_args()
    service = DualSearchService(
        clip_profile=args.clip_profile,
        bge_profile=args.bge_profile,
        image_dir=Path(args.image_dir),
        model_path=args.model_path,
        device=args.device,
        bge_device=args.bge_device,
    )
    app = create_app(service)
    print(f"[dual_search] http://{args.host}:{args.port}/", flush=True)
    print(f"[dual_search] clip={args.clip_profile} bge={args.bge_profile}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
