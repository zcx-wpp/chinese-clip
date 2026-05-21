from __future__ import annotations

import argparse
import json
import mimetypes
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, urlparse

from .hybrid_retrieval import CombinedHybridSearchEngine, HybridSearchConfig, HybridSearchEngine, build_search_engine
from .portable_paths import resolve_portable_path
from .profile_paths import SearchSource, resolve_search_sources


CHUNK_SIZE = 64 * 1024
INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Doubao Video Search</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101114;
      --panel: #181b20;
      --panel-border: #2a2f36;
      --panel-muted: #12151a;
      --text: #edf1f7;
      --muted: #a6afbd;
      --accent: #57c084;
      --accent-strong: #7dd89d;
      --danger: #ff8d7a;
      --shadow: rgba(0, 0, 0, 0.28);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
      background: var(--bg);
      color: var(--text);
    }

    .shell {
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr;
    }

    .toolbar {
      position: sticky;
      top: 0;
      z-index: 10;
      background: rgba(16, 17, 20, 0.92);
      backdrop-filter: blur(16px);
      border-bottom: 1px solid var(--panel-border);
    }

    .toolbar-inner,
    .content {
      width: min(1180px, calc(100vw - 32px));
      margin: 0 auto;
    }

    .toolbar-inner {
      padding: 20px 0 18px;
    }

    h1 {
      margin: 0 0 14px;
      font-size: 22px;
      font-weight: 650;
    }

    .search-row {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 120px;
      gap: 12px;
      align-items: center;
    }

    .search-input,
    .search-button {
      min-height: 48px;
      border-radius: 8px;
      border: 1px solid var(--panel-border);
      font: inherit;
    }

    .search-input {
      padding: 0 16px;
      background: var(--panel);
      color: var(--text);
      outline: none;
    }

    .search-input:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(87, 192, 132, 0.14);
    }

    .search-button {
      cursor: pointer;
      background: var(--accent);
      color: #08110c;
      font-weight: 650;
    }

    .search-button:hover {
      background: var(--accent-strong);
    }

    .status-row {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      margin-top: 12px;
      color: var(--muted);
      font-size: 13px;
      flex-wrap: wrap;
    }

    .status-inline {
      display: flex;
      align-items: center;
      gap: 8px;
      min-height: 20px;
    }

    .dot {
      width: 9px;
      height: 9px;
      border-radius: 999px;
      background: #626b78;
      flex: 0 0 auto;
    }

    .dot.ready { background: var(--accent); }
    .dot.loading { background: #f2c572; }
    .dot.error { background: var(--danger); }

    .content {
      padding: 24px 0 40px;
    }

    .result-count {
      margin-bottom: 16px;
      color: var(--muted);
      font-size: 14px;
    }

    .error-box,
    .empty-box {
      border: 1px solid var(--panel-border);
      background: var(--panel);
      border-radius: 8px;
      padding: 18px 20px;
      color: var(--muted);
    }

    .error-box {
      color: #ffd0c7;
      border-color: rgba(255, 141, 122, 0.4);
    }

    .results {
      display: grid;
      gap: 16px;
    }

    .result-item {
      display: grid;
      grid-template-columns: minmax(280px, 360px) minmax(0, 1fr);
      gap: 18px;
      padding: 18px;
      background: var(--panel);
      border: 1px solid var(--panel-border);
      border-radius: 8px;
      box-shadow: 0 10px 30px var(--shadow);
    }

    .video-frame {
      width: 100%;
      aspect-ratio: 16 / 9;
      border-radius: 8px;
      overflow: hidden;
      background: #050607;
      border: 1px solid #232830;
    }

    video {
      width: 100%;
      height: 100%;
      display: block;
      background: #050607;
    }

    .meta {
      display: grid;
      gap: 10px;
      min-width: 0;
      align-content: start;
    }

    .meta-head {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
    }

    .rank {
      font-size: 18px;
      font-weight: 650;
    }

    .score {
      color: var(--accent-strong);
      font-size: 14px;
      font-variant-numeric: tabular-nums;
    }

    .description {
      color: var(--text);
      line-height: 1.6;
      word-break: break-word;
    }

    .tag-list {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }

    .tag {
      padding: 4px 9px;
      border-radius: 999px;
      background: var(--panel-muted);
      border: 1px solid #2f3640;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.2;
    }

    .kv {
      display: grid;
      grid-template-columns: 92px minmax(0, 1fr);
      gap: 8px;
      align-items: start;
      font-size: 13px;
    }

    .kv-key {
      color: var(--muted);
    }

    .kv-value {
      color: var(--text);
      word-break: break-all;
    }

    .kv-value.mono {
      font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
      font-size: 12px;
    }

    @media (max-width: 860px) {
      .result-item {
        grid-template-columns: 1fr;
      }
    }

    @media (max-width: 640px) {
      .toolbar-inner,
      .content {
        width: min(100vw - 20px, 1180px);
      }

      .search-row {
        grid-template-columns: 1fr;
      }

      .result-item {
        padding: 14px;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <header class="toolbar">
      <div class="toolbar-inner">
        <h1>Doubao Video Search</h1>
        <form id="search-form" class="search-row">
          <input id="query-input" class="search-input" type="text" placeholder="输入中文描述，例如：女子在走廊里搬纸箱" autocomplete="off">
          <button class="search-button" type="submit">搜索 Top 10</button>
        </form>
        <div class="status-row">
          <div id="service-status" class="status-inline"><span class="dot"></span><span>正在连接检索服务</span></div>
          <div id="search-status"></div>
        </div>
      </div>
    </header>

    <main class="content">
      <div id="result-count" class="result-count">等待输入查询。</div>
      <div id="feedback"></div>
      <section id="results" class="results"></section>
    </main>
  </div>

  <script>
    const form = document.getElementById("search-form");
    const input = document.getElementById("query-input");
    const serviceStatus = document.getElementById("service-status");
    const searchStatus = document.getElementById("search-status");
    const resultCount = document.getElementById("result-count");
    const feedback = document.getElementById("feedback");
    const results = document.getElementById("results");

    function setServiceStatus(kind, text) {
      const dotClass = kind === "ready" ? "dot ready" : kind === "error" ? "dot error" : "dot loading";
      serviceStatus.innerHTML = `<span class="${dotClass}"></span><span>${text}</span>`;
    }

    function clearResults() {
      feedback.innerHTML = "";
      results.innerHTML = "";
    }

    function setFeedback(kind, text) {
      const className = kind === "error" ? "error-box" : "empty-box";
      feedback.innerHTML = `<div class="${className}">${text}</div>`;
    }

    function formatSeconds(value) {
      const total = Math.max(0, Math.round(Number(value) || 0));
      const hours = Math.floor(total / 3600);
      const minutes = Math.floor((total % 3600) / 60);
      const seconds = total % 60;
      if (hours > 0) {
        return `${hours}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
      }
      return `${minutes}:${String(seconds).padStart(2, "0")}`;
    }

    function formatElapsedMs(value) {
      const milliseconds = Number(value);
      if (!Number.isFinite(milliseconds) || milliseconds < 0) {
        return "-";
      }
      if (milliseconds >= 1000) {
        return `${(milliseconds / 1000).toFixed(2)} s`;
      }
      return `${milliseconds.toFixed(1)} ms`;
    }

    function createNode(tag, className, text) {
      const node = document.createElement(tag);
      if (className) node.className = className;
      if (text !== undefined) node.textContent = text;
      return node;
    }

    function appendRow(parent, key, value, mono = false) {
      const row = createNode("div", "kv");
      row.appendChild(createNode("div", "kv-key", key));
      row.appendChild(createNode("div", mono ? "kv-value mono" : "kv-value", value));
      parent.appendChild(row);
    }

    function renderResult(item, index) {
      const article = createNode("article", "result-item");

      const media = createNode("div", "video-frame");
      const video = document.createElement("video");
      video.controls = true;
      video.preload = "metadata";
      video.playsInline = true;
      video.src = item.video_url;
      media.appendChild(video);

      const meta = createNode("div", "meta");
      const head = createNode("div", "meta-head");
      head.appendChild(createNode("div", "rank", `#${index + 1} ${item.video_id}`));
      head.appendChild(createNode("div", "score", `融合分 ${Number(item.score || 0).toFixed(4)}`));
      meta.appendChild(head);

      meta.appendChild(createNode("div", "description", item.description || item.caption || "无描述"));

      const tags = Array.isArray(item.tags) ? item.tags : [];
      if (tags.length > 0) {
        const tagList = createNode("div", "tag-list");
        tags.forEach((tag) => tagList.appendChild(createNode("span", "tag", String(tag))));
        meta.appendChild(tagList);
      }

      appendRow(meta, "时长", formatSeconds(item.duration_seconds));
      appendRow(meta, "稀疏排名", item.sparse_rank ? String(item.sparse_rank) : "-");
      appendRow(meta, "稠密排名", item.dense_rank ? String(item.dense_rank) : "-");
      appendRow(meta, "视频路径", item.video_path || "-", true);

      article.appendChild(media);
      article.appendChild(meta);
      return article;
    }

    async function loadStatus() {
      try {
        const response = await fetch("/api/status");
        const payload = await response.json();
        if (payload.ready) {
          setServiceStatus("ready", "检索服务已就绪");
        } else if (payload.starting) {
          setServiceStatus("loading", "检索引擎正在加载");
        } else if (payload.error) {
          setServiceStatus("error", `检索服务未就绪: ${payload.error}`);
        } else {
          setServiceStatus("loading", "检索服务等待初始化");
        }
      } catch (error) {
        setServiceStatus("error", "检索服务状态获取失败");
      }
    }

    // 添加自动轮询状态的函数
    function pollStatus() {
      loadStatus();
      setTimeout(pollStatus, 5000); // 每5秒更新一次状态
    }
    
    // 页面加载完成后开始轮询
    pollStatus();
    
    async function runSearch(event) {
      event.preventDefault();
      const query = input.value.trim();
      if (!query) {
        clearResults();
        resultCount.textContent = "请输入查询文本。";
        return;
      }

      searchStatus.textContent = "检索中...";
      resultCount.textContent = `正在搜索: ${query}`;
      clearResults();

      try {
        const response = await fetch("/api/search", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query, top_k: 10 }),
        });
        const payload = await response.json();

        if (!response.ok) {
          throw new Error(payload.error || "搜索失败");
        }

        const items = Array.isArray(payload.results) ? payload.results : [];
        searchStatus.textContent = "";
        resultCount.textContent = `查询 “${payload.query}” 命中 ${items.length} 条结果`;

        if (items.length === 0) {
          const elapsedText = formatElapsedMs(payload.elapsed_ms);
          resultCount.textContent = `查询 “${payload.query}” 命中 0 条结果，用时 ${elapsedText}`;
          setFeedback("empty", "没有找到匹配视频。可以试试换个描述词，或者补充动作、场景和物体。");
          return;
        }

        const elapsedText = formatElapsedMs(payload.elapsed_ms);
        resultCount.textContent = `查询 “${payload.query}” 命中 ${items.length} 条结果，用时 ${elapsedText}`;
        const fragment = document.createDocumentFragment();
        items.forEach((item, index) => fragment.appendChild(renderResult(item, index)));
        results.appendChild(fragment);
      } catch (error) {
        searchStatus.textContent = "";
        resultCount.textContent = "搜索未完成。";
        setFeedback("error", error.message || "搜索失败");
      } finally {
        loadStatus();
      }
    }

    form.addEventListener("submit", runSearch);
    loadStatus();
  </script>
</body>
</html>
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Hybrid video search UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--profile", help="Optional profile name. Leave empty to search the default merged profile.")
    parser.add_argument("--metadata-db", help="Optional metadata.db path.")
    parser.add_argument("--index-dir", help="Optional hybrid index directory.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--sparse-top-k", type=int, default=100)
    parser.add_argument("--dense-top-k", type=int, default=100)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument(
        "--embedding-device",
        default="cuda",
        help="Embedding device. Defaults to cuda and falls back to cpu when CUDA is unavailable.",
    )
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument(
        "--embedding-local-files-only",
        action="store_true",
        help="Only load the dense embedding model from local files; do not download from Hugging Face.",
    )
    return parser.parse_args()


class SearchAppState:
    def __init__(
        self,
        *,
        sources: list[SearchSource],
        search_config: HybridSearchConfig,
        top_k: int,
        embedding_device: str,
        embedding_batch_size: int,
        embedding_local_files_only: bool,
    ):
        self.sources = list(sources)
        self.search_config = search_config
        self.default_top_k = top_k
        self.embedding_device = embedding_device
        self.embedding_batch_size = embedding_batch_size
        self.embedding_local_files_only = embedding_local_files_only
        self._engine: HybridSearchEngine | CombinedHybridSearchEngine | None = None
        self._last_error: str | None = None
        self._starting = False
        self._lock = threading.Lock()

    def initialize_async(self) -> None:
        thread = threading.Thread(target=self._try_initialize, daemon=True)
        thread.start()

    def _try_initialize(self) -> None:
        with self._lock:
            if self._engine is not None or self._starting:
                return
            self._starting = True
            self._last_error = None

        engine: HybridSearchEngine | CombinedHybridSearchEngine | None = None
        error: str | None = None
        try:
            engine = build_search_engine(
                sources=self.sources,
                search_config=self.search_config,
                embedding_device=self.embedding_device,
                embedding_batch_size=self.embedding_batch_size,
                embedding_local_files_only=self.embedding_local_files_only,
            )
            engine.warmup()
        except Exception as exc:
            error = str(exc)

        with self._lock:
            self._starting = False
            self._engine = engine
            self._last_error = error

    def status_payload(self) -> dict:
        with self._lock:
            ready = self._engine is not None
            starting = self._starting
            error = self._last_error
        return {
            "ready": ready,
            "starting": starting,
            "error": error,
            "profiles": [source.name for source in self.sources],
            "metadata_dbs": [str(source.metadata_db_path) for source in self.sources],
            "index_dirs": [str(source.index_dir) for source in self.sources],
        }

    def require_engine(self) -> HybridSearchEngine | CombinedHybridSearchEngine:
        self._try_initialize()
        with self._lock:
            if self._engine is not None:
                return self._engine
            if self._starting:
                raise RuntimeError("Search engine is still starting. Please retry in a few seconds.")
            raise RuntimeError(self._last_error or "Search engine is unavailable.")

    def search(self, query: str, *, top_k: int) -> tuple[list[dict], float]:
        engine = self.require_engine()
        started_at = time.perf_counter()
        results = engine.search(query, top_k=top_k)
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0
        for item in results:
            raw_path = str(item.get("video_path") or "").strip()
            if raw_path:
                item["video_path"] = str(resolve_portable_path(raw_path))
            item["video_url"] = f"/api/video?video_id={quote(str(item['video_id']), safe='')}"
        return results, elapsed_ms

    def resolve_video_path(self, video_id: str) -> Path:
        engine = self.require_engine()
        row = engine.documents_by_id.get(video_id)
        if row is None:
            raise FileNotFoundError(f"video_id not found: {video_id}")
        raw_path = str(row.get("path") or "").strip()
        if not raw_path:
            raise FileNotFoundError(f"video path missing for {video_id}")
        path = resolve_portable_path(raw_path)
        if not path.exists():
            raise FileNotFoundError(f"video file missing: {path}")
        return path

    def close(self) -> None:
        with self._lock:
            engine = self._engine
            self._engine = None
        if engine is not None:
            engine.close()


class SearchUIHandler(BaseHTTPRequestHandler):
    server: "SearchUIServer"

    def log_message(self, format: str, *args) -> None:
        return

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path.rstrip("/") or "/"
        if route == "/":
            self._send_html(HTTPStatus.OK, INDEX_HTML)
            return
        if route == "/health":
            self._send_json(HTTPStatus.OK, self.server.state.status_payload())
            return
        if route == "/api/status":
            self._send_json(HTTPStatus.OK, self.server.state.status_payload())
            return
        if route == "/api/video":
            params = parse_qs(parsed.query)
            video_id = str((params.get("video_id") or [""])[0]).strip()
            if not video_id:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "video_id is required"})
                return
            try:
                path = self.server.state.resolve_video_path(video_id)
            except FileNotFoundError as exc:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": str(exc)})
                return
            except RuntimeError as exc:
                self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(exc)})
                return
            self._send_file(path)
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/api/search":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return

        try:
            payload = self._read_json()
            query = str(payload.get("query") or "").strip()
            if not query:
                raise ValueError("query is required")
            top_k = int(payload.get("top_k") or self.server.state.default_top_k)
            if top_k <= 0:
                raise ValueError("top_k must be positive")
            results, elapsed_ms = self.server.state.search(query, top_k=top_k)
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        except RuntimeError as exc:
            self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(exc)})
            return
        except Exception as exc:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
            return

        self._send_json(
            HTTPStatus.OK,
            {
                "query": query,
                "count": len(results),
                "elapsed_ms": round(elapsed_ms, 1),
                "results": results,
            },
        )

    def _read_json(self) -> dict:
        content_length = int(self.headers.get("Content-Length") or "0")
        body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
        payload = json.loads(body or "{}")
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _send_html(self, status: HTTPStatus, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: HTTPStatus, payload) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path) -> None:
        file_size = path.stat().st_size
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        range_header = self.headers.get("Range")
        start = 0
        end = file_size - 1
        status = HTTPStatus.OK

        if range_header and range_header.startswith("bytes="):
            spec = range_header.split("=", 1)[1].strip()
            if "," in spec:
                self._send_json(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, {"error": "multiple ranges are not supported"})
                return
            start_text, _, end_text = spec.partition("-")
            try:
                if start_text == "":
                    length = int(end_text)
                    start = max(0, file_size - length)
                else:
                    start = int(start_text)
                if end_text:
                    end = int(end_text)
            except ValueError:
                self._send_json(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, {"error": "invalid range header"})
                return

            if start < 0 or end < start or start >= file_size:
                self._send_json(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE, {"error": "range out of bounds"})
                return
            end = min(end, file_size - 1)
            status = HTTPStatus.PARTIAL_CONTENT

        content_length = end - start + 1
        self.send_response(int(status))
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(content_length))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()

        try:
            with path.open("rb") as handle:
                handle.seek(start)
                remaining = content_length
                while remaining > 0:
                    chunk = handle.read(min(CHUNK_SIZE, remaining))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                        # 客户端连接已断开，停止传输
                        break
                    remaining -= len(chunk)
        except OSError:
            # 文件可能已被移除或不可访问
            pass


class SearchUIServer(ThreadingHTTPServer):
    def __init__(self, server_address, handler_class, *, state: SearchAppState):
        super().__init__(server_address, handler_class)
        self.state = state


def main():
    args = parse_args()
    search_config = HybridSearchConfig(
        sparse_top_k=args.sparse_top_k,
        dense_top_k=args.dense_top_k,
        rrf_k=args.rrf_k,
    )
    sources = resolve_search_sources(
        profile=args.profile,
        metadata_db=args.metadata_db,
        index_dir=args.index_dir,
    )
    state = SearchAppState(
        sources=sources,
        search_config=search_config,
        top_k=args.top_k,
        embedding_device=args.embedding_device,
        embedding_batch_size=args.embedding_batch_size,
        embedding_local_files_only=args.embedding_local_files_only,
    )
    server = SearchUIServer((args.host, args.port), SearchUIHandler, state=state)
    state.initialize_async()
    print(f"[serve] http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] stopping", flush=True)
    finally:
        state.close()
        server.server_close()


if __name__ == "__main__":
    main()
