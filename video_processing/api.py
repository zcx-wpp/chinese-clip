from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

from .config import EmbeddingConfig, RetrievalConfig, VectorStoreConfig
from .embedding import ChineseClipEncoder
from .faiss_store import FaissFrameIndex
from .metadata_store import MetadataStore
from .milvus_store import MilvusFrameIndex
from .retrieval import VideoRetriever


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)


def build_retriever(
    work_dir: Path,
    model_path: str,
    device: str = "cuda",
    vector_backend: str = "faiss",
    milvus_uri: str = "http://127.0.0.1:19530",
    milvus_token: str = "",
    milvus_collection: str = "video_frame_embeddings",
) -> VideoRetriever:
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
    else:
        index = FaissFrameIndex.load(
            index_path=work_dir / "frame_index.faiss",
            meta_path=work_dir / "frame_index.meta.json",
        )
    metadata_store = MetadataStore(work_dir / "metadata.db")
    return VideoRetriever(
        encoder=encoder,
        index=index,
        metadata_store=metadata_store,
        query_templates=embedding_config.query_expansion_templates,
        retrieval_config=RetrievalConfig(),
    )


def create_app(retriever: VideoRetriever) -> FastAPI:
    app = FastAPI(title="Video Search API", version="0.1.0")

    @app.post("/search")
    def search(request: SearchRequest):
        return retriever.search(query=request.query, top_k=request.top_k)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


def parse_args():
    parser = argparse.ArgumentParser(description="Video search API")
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--vector-backend", choices=["faiss", "milvus"], default="faiss")
    parser.add_argument("--milvus-uri", default="http://127.0.0.1:19530")
    parser.add_argument("--milvus-token", default="")
    parser.add_argument("--milvus-collection", default="video_frame_embeddings")
    return parser.parse_args()


def main():
    args = parse_args()
    retriever = build_retriever(
        work_dir=Path(args.work_dir),
        model_path=args.model_path,
        device=args.device,
        vector_backend=args.vector_backend,
        milvus_uri=args.milvus_uri,
        milvus_token=args.milvus_token,
        milvus_collection=args.milvus_collection,
    )
    app = create_app(retriever)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
