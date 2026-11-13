"""Read and normalize source data with page-level traceability and error logging."""
import re
import unicodedata
from pathlib import Path
from typing import Any

import pypdf

from .models import DocumentPage

VALID_CONTENT_PAGES = set(range(6, 48))  # Physical pages 6..47 of NIST AI 100-1


def section_for_page(page_num: int) -> str:
    """Map physical PDF page numbers of NIST AI 100-1 to document sections."""
    if 6 <= page_num <= 8:
        return "executive_summary"
    if 9 <= page_num <= 24:
        return "part1_foundational"
    if 25 <= page_num <= 39:
        return "part2_core_profiles"
    if 40 <= page_num <= 47:
        return "appendix"
    return "excluded"


def normalize_text(text: str) -> str:
    """Normalize Unicode (NFKC), fix hyphenated line breaks, and preserve paragraph breaks."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Rejoin words broken across PDF line wraps (e.g., "envi-\nronment" or "envi- ronment")
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)

    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def load_pdf(
    path: Path,
    document_id: str,
    source: str,
    valid_pages: set[int] | None = VALID_CONTENT_PAGES,
    error_log: list[dict[str, Any]] | None = None,
) -> list[DocumentPage]:
    """Extract each physical page, normalize text, keep page number/source, and log skipped pages."""
    reader = pypdf.PdfReader(str(path))
    pages: list[DocumentPage] = []

    for page_num, raw_page in enumerate(reader.pages, start=1):
        if valid_pages is not None and page_num not in valid_pages:
            if error_log is not None:
                error_log.append(
                    {
                        "document_id": document_id,
                        "page": page_num,
                        "status": "excluded_front_or_back_matter",
                    }
                )
            continue

        try:
            raw_text = raw_page.extract_text() or ""
        except Exception as exc:
            if error_log is not None:
                error_log.append(
                    {
                        "document_id": document_id,
                        "page": page_num,
                        "status": "extraction_error",
                        "error": str(exc),
                    }
                )
            continue

        normalized = normalize_text(raw_text)
        if not normalized:
            if error_log is not None:
                error_log.append(
                    {
                        "document_id": document_id,
                        "page": page_num,
                        "status": "empty_page",
                    }
                )
            continue

        pages.append(
            DocumentPage(
                document_id=document_id,
                source=source,
                page=page_num,
                text=normalized,
                metadata={
                    "section": section_for_page(page_num),
                    "year": 2023,
                    "document_type": "framework",
                    "language": "en",
                },
            )
        )

    return pages
