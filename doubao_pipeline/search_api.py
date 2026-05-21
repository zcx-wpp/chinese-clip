from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .hybrid_retrieval import CombinedHybridSearchEngine, HybridSearchConfig, HybridSearchEngine, build_search_engine
from .profile_paths import resolve_search_sources


def parse_args():
    parser = argparse.ArgumentParser(description="Hybrid video search HTTP API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
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


class SearchHandler(BaseHTTPRequestHandler):
    server: "SearchServer"

    def log_message(self, format: str, *args) -> None:
        return

    def do_GET(self) -> None:
        if self.path.rstrip("/") in {"", "/"}:
            self._send_json(HTTPStatus.OK, {"service": "doubao_pipeline_hybrid_search"})
            return
        if self.path.rstrip("/") == "/health":
            self._send_json(HTTPStatus.OK, {"status": "ok"})
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/search":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return

        try:
            payload = self._read_json()
            query = str(payload.get("query") or "").strip()
            if not query:
                raise ValueError("query is required")
            top_k = int(payload.get("top_k") or self.server.default_top_k)
            sparse_top_k = int(payload.get("sparse_top_k") or self.server.search_config.sparse_top_k)
            dense_top_k = int(payload.get("dense_top_k") or self.server.search_config.dense_top_k)
            rrf_k = int(payload.get("rrf_k") or self.server.search_config.rrf_k)
            results = self.server.engine.search(
                query,
                top_k=top_k,
                sparse_top_k=sparse_top_k,
                dense_top_k=dense_top_k,
                rrf_k=rrf_k,
            )
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        except Exception as exc:
            self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
            return

        self._send_json(HTTPStatus.OK, results)

    def _read_json(self) -> dict:
        content_length = int(self.headers.get("Content-Length") or "0")
        body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else "{}"
        payload = json.loads(body or "{}")
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _send_json(self, status: HTTPStatus, payload) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class SearchServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address,
        handler_class,
        *,
        engine: HybridSearchEngine | CombinedHybridSearchEngine,
        default_top_k: int,
        search_config: HybridSearchConfig,
    ):
        super().__init__(server_address, handler_class)
        self.engine = engine
        self.default_top_k = default_top_k
        self.search_config = search_config


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
    engine = build_search_engine(
        sources=sources,
        search_config=search_config,
        embedding_device=args.embedding_device,
        embedding_batch_size=args.embedding_batch_size,
        embedding_local_files_only=args.embedding_local_files_only,
    )
    engine.warmup()
    server = SearchServer(
        (args.host, args.port),
        SearchHandler,
        engine=engine,
        default_top_k=args.top_k,
        search_config=search_config,
    )
    print(f"[serve] http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] stopping", flush=True)
    finally:
        engine.close()
        server.server_close()


if __name__ == "__main__":
    main()
