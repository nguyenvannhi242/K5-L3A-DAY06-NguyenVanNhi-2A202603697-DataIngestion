"""End-to-end Day 06 Data Ingestion & Retrieval Benchmark pipeline."""
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .chunking import by_paragraph, fixed_size
from .embeddings import MyEmbedder
from .evaluation import evaluate_query, summarize
from .ingestion import load_pdf
from .models import Chunk
from .store import PgVectorStore

REPO_ROOT = Path(__file__).resolve().parents[2]
STARTER_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "results"


def _chunk_stats(chunks: list[Chunk]) -> dict[str, Any]:
    lengths = [len(c.text) for c in chunks]
    return {
        "chunk_count": len(chunks),
        "total_chars": sum(lengths),
        "min_len": min(lengths) if lengths else 0,
        "median_len": int(statistics.median(lengths)) if lengths else 0,
        "max_len": max(lengths) if lengths else 0,
        "avg_len": round(sum(lengths) / len(lengths), 1) if lengths else 0.0,
    }


def main() -> None:
    load_dotenv(STARTER_DIR / ".env")
    model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5").strip() or "BAAI/bge-small-en-v1.5"
    database_url = os.getenv("DATABASE_URL", "").strip()
    if not database_url:
        raise RuntimeError("DATABASE_URL missing in starter/.env. Copy .env.example and start Docker pgvector.")

    pdf_path = STARTER_DIR / "data" / "NIST.AI.100-1.pdf"
    benchmark_path = REPO_ROOT / "benchmark" / "questions.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Ingestion & normalization
    ingestion_log: list[dict[str, Any]] = []
    pages = load_pdf(
        path=pdf_path,
        document_id="nist-ai-100-1",
        source="starter/data/NIST.AI.100-1.pdf",
        error_log=ingestion_log,
    )
    if len(pages) < 30:
        raise RuntimeError(f"Corpus must have >= 30 valid content pages, got {len(pages)}")

    # 2. Load & validate pre-registered benchmark
    benchmark_data = json.loads(benchmark_path.read_text(encoding="utf-8"))
    questions = benchmark_data["questions"]
    dev_count = sum(1 for q in questions if q["split"] == "dev")
    test_count = sum(1 for q in questions if q["split"] == "test")
    if len(questions) < 20 or dev_count < 15 or test_count < 5:
        raise RuntimeError(f"Invalid benchmark split: total={len(questions)}, dev={dev_count}, test={test_count}")

    # 3. Build chunks for Strategies A, B, C (C uses B's chunks, differing only in metadata filter at query time)
    chunks_a = [c for p in pages for c in fixed_size(p, size=500, overlap=50)]
    chunks_b = [c for p in pages for c in by_paragraph(p, max_size=800)]
    chunks_c = [
        Chunk(
            chunk_id=c.chunk_id.replace(":B:", ":C:"),
            document_id=c.document_id,
            source=c.source,
            page=c.page,
            text=c.text,
            metadata=dict(c.metadata),
        )
        for c in chunks_b
    ]

    # 4. Initialize real embedding model and PostgreSQL + pgvector store
    embedder = MyEmbedder(model_name=model_name)
    store = PgVectorStore(database_url=database_url, dimensions=embedder.dimensions)
    store.create_schema()

    strategy_chunks = {"A": chunks_a, "B": chunks_b, "C": chunks_c}
    ingest_metrics: dict[str, Any] = {}

    t0 = time.perf_counter()
    vecs_a = embedder.embed_documents([c.text for c in chunks_a])
    t_embed_a = time.perf_counter() - t0

    t0 = time.perf_counter()
    vecs_b = embedder.embed_documents([c.text for c in chunks_b])
    t_embed_b = time.perf_counter() - t0

    strategy_vectors = {"A": vecs_a, "B": vecs_b, "C": vecs_b}
    embed_times = {"A": t_embed_a, "B": t_embed_b, "C": t_embed_b}

    for strat in ("A", "B", "C"):
        t_up = time.perf_counter()
        store.upsert(strat, strategy_chunks[strat], strategy_vectors[strat])
        upsert_time = time.perf_counter() - t_up
        ingest_metrics[strat] = {
            **_chunk_stats(strategy_chunks[strat]),
            "embed_seconds": round(embed_times[strat], 3),
            "upsert_seconds": round(upsert_time, 3),
            "total_ingest_seconds": round(embed_times[strat] + upsert_time, 3),
            "stored_count": store.count(strat),
        }

    # Idempotency check: re-upsert Strategy A and verify count does not duplicate
    before_reingest = store.count("A")
    store.upsert("A", chunks_a, vecs_a)
    after_reingest = store.count("A")

    # 5. Query & evaluate all 20 questions across A, B, C
    query_vectors = {q["id"]: embedder.embed_query(q["question"]) for q in questions}
    all_traces: list[dict[str, Any]] = []

    for strat in ("A", "B", "C"):
        for q in questions:
            filters = q.get("filter") if strat == "C" else None
            hits = store.search(
                strategy=strat,
                query_vector=query_vectors[q["id"]],
                top_k=3,
                filters=filters,
            )
            all_traces.append(
                evaluate_query(
                    question=q,
                    hits=hits,
                    strategy=strat,
                    filter_applied=filters,
                )
            )

    summary_rows = summarize(all_traces)

    # 6. Prepare deterministic samples for manual assessment:
    # - Coherence: first 10 sorted valid pages (P06..P15), matching chunks covering the opening of each page
    coherence_pages = sorted(p.page for p in pages)[:10]
    coherence_samples: list[dict[str, Any]] = []
    for pnum in coherence_pages:
        page_obj = next(p for p in pages if p.page == pnum)
        first_chunk_a = next((c for c in chunks_a if c.page == pnum), None)
        first_chunk_b = next((c for c in chunks_b if c.page == pnum), None)
        coherence_samples.append(
            {
                "page": pnum,
                "section": page_obj.metadata.get("section"),
                "source_opening": page_obj.text[:300],
                "A_chunk": {
                    "chunk_id": first_chunk_a.chunk_id if first_chunk_a else None,
                    "text": first_chunk_a.text if first_chunk_a else "",
                },
                "B_and_C_chunk": {
                    "chunk_id": first_chunk_b.chunk_id if first_chunk_b else None,
                    "text": first_chunk_b.text if first_chunk_b else "",
                },
            }
        )

    test_traces = [t for t in all_traces if t["split"] == "test"]

    # 7. Save artifacts to results/
    (RESULTS_DIR / "traces.json").write_text(
        json.dumps(all_traces, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (RESULTS_DIR / "coherence_samples.json").write_text(
        json.dumps(coherence_samples, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (RESULTS_DIR / "test_audit_samples.json").write_text(
        json.dumps(test_traces, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary_payload = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "student": "Nguyễn Văn Nhi (2A202603697 - L3A)",
        "embedding_model": embedder.model_name,
        "embedding_dimensions": embedder.dimensions,
        "vector_store": "PostgreSQL 16 + pgvector 0.8.7 (Docker local, cosine <=>, exact)",
        "cost_usd": 0.0,
        "valid_pages_count": len(pages),
        "skipped_or_logged_pages": ingestion_log,
        "idempotency_check_strategy_A": {
            "before_reingest": before_reingest,
            "after_reingest": after_reingest,
            "no_duplicates": before_reingest == after_reingest,
        },
        "ingest_metrics": ingest_metrics,
        "retrieval_summary": summary_rows,
    }
    (RESULTS_DIR / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("=== Day 06 Pipeline Complete ===")
    print(f"Model: {embedder.model_name} ({embedder.dimensions}d) | Cost: $0.00")
    print(f"Valid pages: {len(pages)} | Idempotency A: {before_reingest} -> {after_reingest}")
    for row in summary_rows:
        print(
            f"Strategy {row['strategy']} [{row['split']:4s}]: "
            f"Hit@1={row['hit_at_1_count']}/{row['total']} ({row['hit_at_1_rate']:.1%}) | "
            f"Hit@3={row['hit_at_3_count']}/{row['total']} ({row['hit_at_3_rate']:.1%}) | "
            f"Filter={row['filter_compliance']}"
        )


if __name__ == "__main__":
    main()
