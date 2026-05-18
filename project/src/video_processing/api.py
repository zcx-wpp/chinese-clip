from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from .config import EmbeddingConfig, PROJECT_ROOT, RetrievalConfig, VectorStoreConfig
from .profile_paths import default_metadata_db_path, default_output_dir, resolve_path

if TYPE_CHECKING:
    from .faiss_store import FaissFrameIndex
    from fastapi import FastAPI
    from .retrieval import VideoRetriever


from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)


class SegmentResult(BaseModel):
    start: float
    end: float
    score: float


class SearchResult(BaseModel):
    video_id: str
    score: float
    segments: list[SegmentResult]
    video_path: str


def _load_optional_faiss_index(index_path: Path, meta_path: Path):
    if not index_path.exists() or not meta_path.exists():
        return None
    from .faiss_store import FaissFrameIndex

    return FaissFrameIndex.load(index_path=index_path, meta_path=meta_path)


def _resolve_runtime_paths(args) -> tuple[Path, Path]:
    return (
        resolve_path(args.output_dir, default_output_dir(args.profile)),
        resolve_path(args.metadata_db, default_metadata_db_path(args.profile)),
    )


def build_retriever(
    output_dir: Path,
    metadata_db_path: Path,
    model_path: str,
    device: str = "cuda",
    retrieval_preset: str = "current",
    video_recall_top_k: int | None = None,
    segment_recall_top_k: int | None = None,
    video_recall_candidate_pool_size: int | None = None,
    segment_recall_candidate_pool_size: int | None = None,
    rerank_score_agg_mode: str | None = None,
    rerank_top_k_average: int | None = None,
    rerank_smoothmax_beta: float | None = None,
    clip_score_weight: float | None = None,
    motion_score_weight: float | None = None,
    rerank_segment_support_weight: float | None = None,
    rerank_genericness_penalty_weight: float | None = None,
    vector_backend: str = "faiss",
    milvus_uri: str = "http://127.0.0.1:19530",
    milvus_token: str = "",
    milvus_collection: str = "video_frame_embeddings",
) -> "VideoRetriever":
    from .embedding import ChineseClipEncoder
    from .metadata_store import MetadataStore
    from .milvus_store import MilvusFrameIndex
    from .retrieval import VideoRetriever

    embedding_config = EmbeddingConfig(model_path=model_path, device=device)
    vector_config = VectorStoreConfig(
        backend=vector_backend,
        milvus_uri=milvus_uri,
        milvus_token=milvus_token,
        milvus_collection=milvus_collection,
    )
    encoder = ChineseClipEncoder(
        model_path=embedding_config.model_path,
        device=embedding_config.device,
        batch_size=embedding_config.batch_size,
    )
    if vector_config.backend == "milvus":
        index = MilvusFrameIndex(
            uri=vector_config.milvus_uri,
            token=vector_config.milvus_token,
            collection_name=vector_config.milvus_collection,
        )
        segment_index = index
        video_index = None
    else:
        from .faiss_store import FaissFrameIndex

        index = FaissFrameIndex.load(
            index_path=output_dir / "faiss" / "frame_index.faiss",
            meta_path=output_dir / "faiss" / "frame_index.meta.json",
        )
        segment_index = _load_optional_faiss_index(
            index_path=output_dir / "faiss" / "segment_index.faiss",
            meta_path=output_dir / "faiss" / "segment_index.meta.json",
        )
        if segment_index is None:
            segment_index = index

        video_index = _load_optional_faiss_index(
            index_path=output_dir / "faiss" / "video_index.faiss",
            meta_path=output_dir / "faiss" / "video_index.meta.json",
        )
    metadata_store = MetadataStore(metadata_db_path)
    retrieval_config = RetrievalConfig.for_preset(retrieval_preset)
    if video_recall_top_k is not None:
        retrieval_config.video_recall_top_k = video_recall_top_k
    if segment_recall_top_k is not None:
        retrieval_config.segment_recall_top_k = segment_recall_top_k
    if video_recall_candidate_pool_size is not None:
        retrieval_config.video_recall_candidate_pool_size = video_recall_candidate_pool_size
    if segment_recall_candidate_pool_size is not None:
        retrieval_config.segment_recall_candidate_pool_size = segment_recall_candidate_pool_size
    if rerank_score_agg_mode is not None:
        retrieval_config.rerank_score_agg_mode = rerank_score_agg_mode
    if rerank_top_k_average is not None:
        retrieval_config.rerank_top_k_average = rerank_top_k_average
    if rerank_smoothmax_beta is not None:
        retrieval_config.rerank_smoothmax_beta = rerank_smoothmax_beta
    if clip_score_weight is not None:
        retrieval_config.clip_score_weight = clip_score_weight
    if motion_score_weight is not None:
        retrieval_config.motion_score_weight = motion_score_weight
    if rerank_segment_support_weight is not None:
        retrieval_config.rerank_segment_support_weight = rerank_segment_support_weight
    if rerank_genericness_penalty_weight is not None:
        retrieval_config.rerank_genericness_penalty_weight = rerank_genericness_penalty_weight
    return VideoRetriever(
        encoder=encoder,
        index=index,
        segment_index=segment_index,
        video_index=video_index,
        metadata_store=metadata_store,
        retrieval_config=retrieval_config,
    )


def create_app(retriever: "VideoRetriever") -> "FastAPI":
    from fastapi import FastAPI
    from fastapi.responses import FileResponse, HTMLResponse

    def render_home_page() -> str:
        return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Video Search</title>
  <style>
    :root {
      color-scheme: light dark;
      --bg: #0f1115;
      --panel: #171a21;
      --panel-2: #1e232d;
      --text: #f3f6fb;
      --muted: #aeb8c8;
      --accent: #67b3ff;
      --border: #2a3240;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "Segoe UI", Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    main {
      max-width: 1400px;
      margin: 0 auto;
      padding: 24px;
    }
    .toolbar {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 12px;
      margin-bottom: 20px;
    }
    input[type="text"] {
      width: 100%;
      min-height: 48px;
      padding: 0 14px;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--panel);
      color: var(--text);
      font-size: 16px;
    }
    button {
      min-width: 120px;
      min-height: 48px;
      border: 0;
      border-radius: 8px;
      background: var(--accent);
      color: #04111f;
      font-size: 15px;
      font-weight: 600;
      cursor: pointer;
    }
    button:disabled {
      cursor: wait;
      opacity: 0.7;
    }
    .status {
      min-height: 22px;
      color: var(--muted);
      margin-bottom: 16px;
      font-size: 14px;
    }
    .results {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
      gap: 16px;
    }
    .result {
      border: 1px solid var(--border);
      border-radius: 8px;
      overflow: hidden;
      background: var(--panel);
    }
    video {
      display: block;
      width: 100%;
      aspect-ratio: 16 / 9;
      background: #000;
    }
    .meta {
      padding: 12px;
      display: grid;
      gap: 8px;
    }
    .title {
      font-size: 14px;
      font-weight: 600;
      word-break: break-all;
    }
    .sub {
      color: var(--muted);
      font-size: 13px;
    }
    .segments {
      display: grid;
      gap: 6px;
      font-size: 12px;
      color: var(--muted);
    }
  </style>
</head>
<body>
  <main>
    <div class="toolbar">
      <input id="query" type="text" placeholder="输入中文描述，比如：一个女人坐在沙滩上把桶里的沙子倒出来">
      <button id="searchBtn" type="button">搜索 Top 10</button>
    </div>
    <div id="status" class="status">输入文本后开始搜索。</div>
    <section id="results" class="results"></section>
  </main>
  <script>
    const queryInput = document.getElementById("query");
    const searchBtn = document.getElementById("searchBtn");
    const statusEl = document.getElementById("status");
    const resultsEl = document.getElementById("results");

    function formatSegment(segment) {
      return `${segment.start.toFixed(2)}s - ${segment.end.toFixed(2)}s  score=${segment.score.toFixed(4)}`;
    }

    function renderResults(items) {
      resultsEl.innerHTML = "";
      for (const [index, item] of items.entries()) {
        const card = document.createElement("article");
        card.className = "result";
        const segments = (item.segments || []).map(formatSegment).join("</div><div>");
        const mediaUrl = `/media?path=${encodeURIComponent(item.video_path)}`;
        card.innerHTML = `
          <video controls preload="metadata" src="${mediaUrl}"></video>
          <div class="meta">
            <div class="title">#${index + 1} ${item.video_id}</div>
            <div class="sub">score=${item.score.toFixed(4)}</div>
            <div class="segments"><div>${segments || "无片段信息"}</div></div>
          </div>
        `;
        resultsEl.appendChild(card);
      }
    }

    async function runSearch() {
      const query = queryInput.value.trim();
      if (!query) {
        statusEl.textContent = "先输入一点文本。";
        queryInput.focus();
        return;
      }
      searchBtn.disabled = true;
      statusEl.textContent = "正在检索，请稍等...";
      resultsEl.innerHTML = "";
      try {
        const response = await fetch("/search", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query, top_k: 10 }),
        });
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`);
        }
        const payload = await response.json();
        statusEl.textContent = `找到 ${payload.length} 条结果。`;
        renderResults(payload);
      } catch (error) {
        statusEl.textContent = `检索失败：${error.message}`;
      } finally {
        searchBtn.disabled = false;
      }
    }

    searchBtn.addEventListener("click", runSearch);
    queryInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter") {
        runSearch();
      }
    });
  </script>
</body>
</html>"""

    app = FastAPI(title="Video Search API", version="0.1.0")

    @app.get("/", response_class=HTMLResponse)
    def home():
        return render_home_page()

    @app.post("/search", response_model=list[SearchResult])
    def search(request: SearchRequest):
        return retriever.search(query=request.query, top_k=request.top_k)

    @app.get("/media")
    def media(path: str):
        file_path = Path(path)
        if not file_path.exists() or not file_path.is_file():
            return HTMLResponse("File not found", status_code=404)
        return FileResponse(file_path)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


def parse_args():
    parser = argparse.ArgumentParser(description="Video search API")
    parser.add_argument("--output-dir")
    parser.add_argument("--metadata-db")
    parser.add_argument("--profile", help="Named storage profile for side-by-side indexes, e.g. seg4s.")
    parser.add_argument("--model-path", default=str(PROJECT_ROOT / "models"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--retrieval-preset", choices=["current", "baseline"], default="current")
    parser.add_argument("--video-recall-top-k", type=int, help="Optional override for video recall search count.")
    parser.add_argument("--segment-recall-top-k", type=int, help="Optional override for segment recall search count.")
    parser.add_argument(
        "--video-recall-candidate-pool-size",
        type=int,
        help="Optional override for video recall candidate pool size.",
    )
    parser.add_argument(
        "--segment-recall-candidate-pool-size",
        type=int,
        help="Optional override for segment recall candidate pool size.",
    )
    parser.add_argument(
        "--rerank-score-agg-mode",
        choices=["topk_average", "smoothmax", "consensus_smoothmax"],
        help="Optional override for final video score aggregation mode.",
    )
    parser.add_argument("--rerank-top-k-average", type=int, help="Optional override for rerank top-k aggregation count.")
    parser.add_argument("--rerank-smoothmax-beta", type=float, help="Optional override for rerank smoothmax beta.")
    parser.add_argument("--clip-score-weight", type=float, help="Optional override for clip score weight.")
    parser.add_argument("--motion-score-weight", type=float, help="Optional override for motion score weight.")
    parser.add_argument(
        "--rerank-segment-support-weight",
        type=float,
        help="Optional override for segment support weight.",
    )
    parser.add_argument(
        "--rerank-genericness-penalty-weight",
        type=float,
        help="Optional override for genericness penalty weight.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--vector-backend", choices=["faiss", "milvus"], default="faiss")
    parser.add_argument("--milvus-uri", default="http://127.0.0.1:19530")
    parser.add_argument("--milvus-token", default="")
    parser.add_argument("--milvus-collection", default="video_frame_embeddings")
    return parser.parse_args()


def main():
    import uvicorn

    args = parse_args()
    output_dir, metadata_db_path = _resolve_runtime_paths(args)
    retriever = build_retriever(
        output_dir=output_dir,
        metadata_db_path=metadata_db_path,
        model_path=args.model_path,
        device=args.device,
        retrieval_preset=args.retrieval_preset,
        video_recall_top_k=args.video_recall_top_k,
        segment_recall_top_k=args.segment_recall_top_k,
        video_recall_candidate_pool_size=args.video_recall_candidate_pool_size,
        segment_recall_candidate_pool_size=args.segment_recall_candidate_pool_size,
        rerank_score_agg_mode=args.rerank_score_agg_mode,
        rerank_top_k_average=args.rerank_top_k_average,
        rerank_smoothmax_beta=args.rerank_smoothmax_beta,
        clip_score_weight=args.clip_score_weight,
        motion_score_weight=args.motion_score_weight,
        rerank_segment_support_weight=args.rerank_segment_support_weight,
        rerank_genericness_penalty_weight=args.rerank_genericness_penalty_weight,
        vector_backend=args.vector_backend,
        milvus_uri=args.milvus_uri,
        milvus_token=args.milvus_token,
        milvus_collection=args.milvus_collection,
    )
    app = create_app(retriever)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
