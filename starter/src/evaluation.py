"""Evaluate retrieved evidence and preserve per-query traces."""
import re
import unicodedata
from typing import Any


def _norm_for_match(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def _is_evidence_hit(question: dict[str, Any], hit: dict[str, Any]) -> bool:
    """Return True if hit matches gold source/page and contains the gold evidence quote."""
    if hit.get("document_id") != question["gold_source_id"]:
        return False
    if int(hit.get("page", -1)) != int(question["gold_page"]):
        return False
    gold_ev = _norm_for_match(question["evidence_quote"])
    chunk_txt = _norm_for_match(hit.get("text", ""))
    return gold_ev in chunk_txt


def _is_filter_compliant(filters: dict[str, Any] | None, hits: list[dict[str, Any]]) -> bool | None:
    """Return True if all hits satisfy filters, False if any violates, or None if no filter."""
    if not filters:
        return None
    for hit in hits:
        meta = hit.get("metadata", {})
        for k, v in filters.items():
            actual = hit.get(k, meta.get(k))
            if actual != v:
                return False
    return True


def evaluate_query(
    question: dict[str, Any],
    hits: list[dict[str, Any]],
    strategy: str,
    filter_applied: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compute Hit@1, Evidence Hit@3, and filter compliance while preserving full top-3 trace."""
    hit_flags = [_is_evidence_hit(question, h) for h in hits]
    hit_at_1 = bool(hit_flags[0]) if hit_flags else False
    hit_at_3 = any(hit_flags[:3])
    filter_compliant = _is_filter_compliant(filter_applied, hits[:3])

    return {
        "strategy": strategy,
        "id": question["id"],
        "split": question["split"],
        "type": question["type"],
        "question": question["question"],
        "expected_answer": question["expected_answer"],
        "gold_source_id": question["gold_source_id"],
        "gold_page": question["gold_page"],
        "evidence_quote": question["evidence_quote"],
        "filter_requested": question.get("filter"),
        "filter_applied": filter_applied,
        "hit_at_1": hit_at_1,
        "hit_at_3": hit_at_3,
        "per_rank_hit": hit_flags[:3],
        "filter_compliant": filter_compliant,
        "top_3_hits": hits[:3],
    }


def summarize(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate Hit@1, Evidence Hit@3, and filter compliance by strategy and split."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for t in traces:
        groups.setdefault((t["strategy"], t["split"]), []).append(t)

    summary_rows: list[dict[str, Any]] = []
    for (strategy, split), items in sorted(groups.items()):
        n = len(items)
        h1 = sum(1 for x in items if x["hit_at_1"])
        h3 = sum(1 for x in items if x["hit_at_3"])
        filtered = [x for x in items if x["filter_applied"] is not None]
        if filtered:
            fc_ok = sum(1 for x in filtered if x["filter_compliant"] is True)
            fc_str = f"{fc_ok}/{len(filtered)}"
        else:
            fc_str = "N/A"

        summary_rows.append(
            {
                "strategy": strategy,
                "split": split,
                "total": n,
                "hit_at_1_count": h1,
                "hit_at_1_rate": round(h1 / n, 4) if n else 0.0,
                "hit_at_3_count": h3,
                "hit_at_3_rate": round(h3 / n, 4) if n else 0.0,
                "filter_compliance": fc_str,
            }
        )
    return summary_rows
