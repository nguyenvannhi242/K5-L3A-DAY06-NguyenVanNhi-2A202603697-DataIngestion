"""Store and search vectors using PostgreSQL + pgvector with cosine distance and metadata filters."""
from typing import Any

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from .models import Chunk

ALLOWED_FILTER_KEYS = {"section", "document_id", "year", "document_type", "language"}


def _vector_literal(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


class PgVectorStore:
    """PostgreSQL + pgvector store supporting idempotent upserts and parameterized SQL filters."""

    def __init__(self, database_url: str, dimensions: int) -> None:
        if not database_url:
            raise ValueError("database_url must not be empty")
        if dimensions <= 0:
            raise ValueError(f"dimensions must be positive, got {dimensions}")
        self.database_url = database_url
        self.dimensions = dimensions

    def create_schema(self) -> None:
        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS chunks (
                        strategy TEXT NOT NULL,
                        chunk_id TEXT NOT NULL,
                        document_id TEXT NOT NULL,
                        source TEXT NOT NULL,
                        page INTEGER NOT NULL,
                        section TEXT NOT NULL,
                        year INTEGER,
                        document_type TEXT,
                        language TEXT,
                        text TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        embedding vector({self.dimensions}) NOT NULL,
                        PRIMARY KEY (strategy, chunk_id)
                    );
                    """
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_chunks_strategy_section ON chunks (strategy, section);"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_chunks_strategy_doc ON chunks (strategy, document_id);"
                )
            conn.commit()

    def count(self, strategy: str) -> int:
        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM chunks WHERE strategy = %s;", (strategy,))
                row = cur.fetchone()
                return int(row[0]) if row else 0

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"Chunk count ({len(chunks)}) != vector count ({len(vectors)})")
        for idx, vec in enumerate(vectors):
            if len(vec) != self.dimensions:
                raise ValueError(
                    f"Vector {idx} has dimension {len(vec)}, expected {self.dimensions}"
                )

        self.create_schema()

        incoming_ids_by_doc: dict[str, list[str]] = {}
        for chunk in chunks:
            incoming_ids_by_doc.setdefault(chunk.document_id, []).append(chunk.chunk_id)

        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                for doc_id, keep_ids in incoming_ids_by_doc.items():
                    cur.execute(
                        """
                        DELETE FROM chunks
                        WHERE strategy = %s
                          AND document_id = %s
                          AND NOT (chunk_id = ANY(%s));
                        """,
                        (strategy, doc_id, keep_ids),
                    )

                for chunk, vec in zip(chunks, vectors):
                    meta = dict(chunk.metadata)
                    cur.execute(
                        """
                        INSERT INTO chunks (
                            strategy, chunk_id, document_id, source, page,
                            section, year, document_type, language,
                            text, metadata, embedding
                        )
                        VALUES (
                            %s, %s, %s, %s, %s,
                            %s, %s, %s, %s,
                            %s, %s, %s::vector
                        )
                        ON CONFLICT (strategy, chunk_id) DO UPDATE SET
                            document_id = EXCLUDED.document_id,
                            source = EXCLUDED.source,
                            page = EXCLUDED.page,
                            section = EXCLUDED.section,
                            year = EXCLUDED.year,
                            document_type = EXCLUDED.document_type,
                            language = EXCLUDED.language,
                            text = EXCLUDED.text,
                            metadata = EXCLUDED.metadata,
                            embedding = EXCLUDED.embedding;
                        """,
                        (
                            strategy,
                            chunk.chunk_id,
                            chunk.document_id,
                            chunk.source,
                            chunk.page,
                            str(meta.get("section", "")),
                            meta.get("year"),
                            meta.get("document_type"),
                            meta.get("language"),
                            chunk.text,
                            Jsonb(meta),
                            _vector_literal(vec),
                        ),
                    )
            conn.commit()

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

        where_clauses = [sql.SQL("strategy = %s")]
        filter_params: list[Any] = [strategy]

        if filters:
            for key, value in filters.items():
                if key not in ALLOWED_FILTER_KEYS:
                    raise ValueError(f"Unsupported filter key: {key}")
                where_clauses.append(sql.SQL("{} = %s").format(sql.Identifier(key)))
                filter_params.append(value)

        qvec_lit = _vector_literal(query_vector)
        query = sql.SQL(
            """
            SELECT chunk_id, document_id, source, page, text, metadata,
                   (embedding <=> %s::vector) AS distance
            FROM chunks
            WHERE {where_sql}
            ORDER BY embedding <=> %s::vector ASC
            LIMIT %s;
            """
        ).format(where_sql=sql.SQL(" AND ").join(where_clauses))

        params = [qvec_lit, *filter_params, qvec_lit, top_k]

        with psycopg.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()

        results: list[dict[str, Any]] = []
        for chunk_id, doc_id, source, page, text, metadata, distance in rows:
            results.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": doc_id,
                    "source": source,
                    "page": int(page),
                    "text": text,
                    "score": round(1.0 - float(distance), 6),
                    "metadata": dict(metadata or {}),
                }
            )
        return results
