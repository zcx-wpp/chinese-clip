from __future__ import annotations

import hashlib

import numpy as np

from .vector_store import VectorStore


class MilvusFrameIndex(VectorStore):
    def __init__(
        self,
        uri: str,
        token: str,
        collection_name: str,
        dim: int = 512,
        index_type: str = "HNSW",
        metric_type: str = "IP",
        m: int = 16,
        ef_construction: int = 200,
    ):
        try:
            from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, MilvusClient, connections, utility
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "pymilvus is not installed. Install it with: pip install pymilvus"
            ) from exc

        self._Collection = Collection
        self._connections = connections
        self._utility = utility
        self._MilvusClient = MilvusClient
        self.uri = uri
        self.token = token
        self.collection_name = collection_name
        self.dim = dim
        self.metric_type = metric_type
        self.index_type = index_type
        self.index_params = {"M": m, "efConstruction": ef_construction}

        self._connections.connect(alias="default", uri=uri, token=token or None)
        if not self._utility.has_collection(collection_name):
            fields = [
                FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
                FieldSchema(name="frame_id", dtype=DataType.VARCHAR, max_length=256),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            ]
            schema = CollectionSchema(fields=fields, description="Video frame embeddings")
            collection = Collection(name=collection_name, schema=schema)
            collection.create_index(
                field_name="embedding",
                index_params={
                    "index_type": index_type,
                    "metric_type": metric_type,
                    "params": self.index_params,
                },
            )
        self.collection = self._Collection(collection_name)
        self.collection.load()

    def _pk(self, frame_id: str) -> int:
        return int(hashlib.md5(frame_id.encode("utf-8")).hexdigest()[:15], 16)

    def add(self, frame_ids: list[str], embeddings: np.ndarray):
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
        data = [
            [self._pk(frame_id) for frame_id in frame_ids],
            frame_ids,
            embeddings.tolist(),
        ]
        self.collection.insert(data)
        self.collection.flush()

    def search(self, query_embeddings: np.ndarray, top_k: int) -> list[list[tuple[str, float]]]:
        if query_embeddings.dtype != np.float32:
            query_embeddings = query_embeddings.astype(np.float32)
        search_params = {"metric_type": self.metric_type, "params": {"ef": max(64, top_k)}}
        results = self.collection.search(
            data=query_embeddings.tolist(),
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["frame_id"],
        )
        output = []
        for result in results:
            row = []
            for hit in result:
                row.append((hit.entity.get("frame_id"), float(hit.score)))
            output.append(row)
        return output

    def save(self):
        self.collection.flush()
