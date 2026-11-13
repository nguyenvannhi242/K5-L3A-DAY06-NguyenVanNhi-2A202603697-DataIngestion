"""Persistent local vector store using Qdrant with cosine similarity and metadata filtering."""
import uuid
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    VectorParams,
)

from .models import Chunk

ALLOWED_FILTER_KEYS = {"section", "document_id", "year", "document_type", "language"}


class QdrantVectorStore:
    """Local persistent Qdrant vector store supporting idempotent upserts and metadata filters."""

    def __init__(self, storage_path: Path | str, dimensions: int) -> None:
        if dimensions <= 0:
            raise ValueError(f"dimensions must be positive, got {dimensions}")
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.dimensions = dimensions
        self.client = QdrantClient(path=str(self.storage_path))

    @staticmethod
    def _collection_name(strategy: str) -> str:
        return f"day6_{strategy.lower()}"

    @staticmethod
    def _point_id(chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    def create_schema(self, strategy: str) -> None:
        name = self._collection_name(strategy)
        if not self.client.collection_exists(name):
            self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=self.dimensions, distance=Distance.COSINE),
            )

    def count(self, strategy: str) -> int:
        name = self._collection_name(strategy)
        if not self.client.collection_exists(name):
            return 0
        return int(self.client.count(collection_name=name, exact=True).count)

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"Chunk count ({len(chunks)}) != vector count ({len(vectors)})")
        for idx, vec in enumerate(vectors):
            if len(vec) != self.dimensions:
                raise ValueError(
                    f"Vector {idx} has dimension {len(vec)}, expected {self.dimensions}"
                )

        self.create_schema(strategy)
        name = self._collection_name(strategy)

        # Remove obsolete chunks for re-ingested document_ids to prevent stale duplicates
        incoming_ids_by_doc: dict[str, set[str]] = {}
        for chunk in chunks:
            incoming_ids_by_doc.setdefault(chunk.document_id, set()).add(chunk.chunk_id)

        for doc_id, keep_chunk_ids in incoming_ids_by_doc.items():
            existing, _ = self.client.scroll(
                collection_name=name,
                scroll_filter=Filter(
                    must=[FieldCondition(key="document_id", match=MatchValue(value=doc_id))]
                ),
                limit=10000,
                with_payload=["chunk_id"],
                with_vectors=False,
            )
            obsolete_point_ids = [
                pt.id
                for pt in existing
                if pt.payload and pt.payload.get("chunk_id") not in keep_chunk_ids
            ]
            if obsolete_point_ids:
                self.client.delete(
                    collection_name=name,
                    points_selector=PointIdsList(points=obsolete_point_ids),
                )

        points = [
            PointStruct(
                id=self._point_id(chunk.chunk_id),
                vector=vec,
                payload={
                    "chunk_id": chunk.chunk_id,
                    "document_id": chunk.document_id,
                    "source": chunk.source,
                    "page": chunk.page,
                    "text": chunk.text,
                    **chunk.metadata,
                },
            )
            for chunk, vec in zip(chunks, vectors)
        ]

        if points:
            self.client.upsert(collection_name=name, points=points, wait=True)

    def search(
        self,
        strategy: str,
        query_vector: list[float],
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if len(query_vector) != self.dimensions:
            raise ValueError(
                f"Query vector dimension {len(query_vector)} != expected {self.dimensions}"
            )

        qdrant_filter: Filter | None = None
        if filters:
            conditions = []
            for key, value in filters.items():
                if key not in ALLOWED_FILTER_KEYS:
                    raise ValueError(f"Unsupported filter key: {key}")
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
            qdrant_filter = Filter(must=conditions)

        name = self._collection_name(strategy)
        response = self.client.query_points(
            collection_name=name,
            query=query_vector,
            query_filter=qdrant_filter,
            limit=top_k,
            with_payload=True,
        )

        results: list[dict[str, Any]] = []
        for point in response.points:
            payload = dict(point.payload or {})
            results.append(
                {
                    "chunk_id": payload.get("chunk_id"),
                    "document_id": payload.get("document_id"),
                    "source": payload.get("source"),
                    "page": payload.get("page"),
                    "text": payload.get("text", ""),
                    "score": round(float(point.score), 6),
                    "metadata": {
                        k: v
                        for k, v in payload.items()
                        if k not in {"chunk_id", "document_id", "source", "page", "text"}
                    },
                }
            )
        return results
