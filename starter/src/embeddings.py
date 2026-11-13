"""Real local embedding adapter using SentenceTransformers (no mock fallback)."""
from typing import Protocol

from sentence_transformers import SentenceTransformer


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class MyEmbedder:
    """SentenceTransformer embedding adapter; raises on failure without mock fallback."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5") -> None:
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        dim = self._model.get_sentence_embedding_dimension()
        if dim is None:
            raise RuntimeError(f"Could not determine embedding dimensions for {model_name}")
        self.dimensions = int(dim)
        self._query_prefix = (
            "Represent this sentence for searching relevant passages: "
            if "bge-" in model_name.lower()
            else ""
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [vec.tolist() for vec in vectors]

    def embed_query(self, question: str) -> list[float]:
        query_text = f"{self._query_prefix}{question}" if self._query_prefix else question
        vec = self._model.encode(
            query_text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vec.tolist()
