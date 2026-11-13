"""Implement fixed-size and structure-aware (paragraph/heading) chunking."""
import re
from .models import Chunk, DocumentPage


def _flatten(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def fixed_size(page: DocumentPage, size: int = 500, overlap: int = 50) -> list[Chunk]:
    """Split normalized page text into overlapping fixed-size character windows."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    flat = _flatten(page.text)
    if not flat:
        return []

    chunks: list[Chunk] = []
    step = size - overlap
    start = 0
    idx = 0

    while start < len(flat):
        window = flat[start : start + size].strip()
        if window:
            meta = dict(page.metadata)
            meta["chunk_index"] = idx
            chunks.append(
                Chunk(
                    chunk_id=f"{page.document_id}:A:p{page.page:02d}:c{idx:03d}",
                    document_id=page.document_id,
                    source=page.source,
                    page=page.page,
                    text=window,
                    metadata=meta,
                )
            )
            idx += 1
        if start + size >= len(flat):
            break
        start += step

    return chunks


def _split_paragraphs(text: str) -> list[str]:
    """Split page text into logical paragraphs/blocks using blank lines or heading/bullet cues."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    if len(blocks) > 1:
        return [_flatten(b) for b in blocks]

    # Fallback for PDFs without double newlines: split at sentence-ending lines or bullet/section headers
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    paragraphs: list[str] = []
    buf: list[str] = []

    for line in lines:
        is_header_or_bullet = bool(
            re.match(r"^(•|\d+\.\d*\.?\s|[A-Z]{2,}\s+\d+\.\d+:?|Table\s+\d+|Appendix\s+[A-D])", line)
        )
        if is_header_or_bullet and buf:
            paragraphs.append(_flatten(" ".join(buf)))
            buf = [line]
        else:
            buf.append(line)
            if line.endswith((".", "?", "!")) and len(" ".join(buf)) >= 350:
                paragraphs.append(_flatten(" ".join(buf)))
                buf = []

    if buf:
        paragraphs.append(_flatten(" ".join(buf)))
    return [p for p in paragraphs if p]


def _split_oversized(para: str, max_size: int, overlap: int = 50) -> list[str]:
    """Split an oversized paragraph at sentence boundaries when possible."""
    if len(para) <= max_size:
        return [para]

    sentences = re.split(r"(?<=[.!?])\s+", para)
    parts: list[str] = []
    current = ""

    for sent in sentences:
        if len(sent) > max_size:
            if current:
                parts.append(current.strip())
                current = ""
            start = 0
            step = max_size - overlap
            while start < len(sent):
                parts.append(sent[start : start + max_size].strip())
                if start + max_size >= len(sent):
                    break
                start += step
        elif not current:
            current = sent
        elif len(current) + 1 + len(sent) <= max_size:
            current = f"{current} {sent}"
        else:
            parts.append(current.strip())
            current = sent

    if current:
        parts.append(current.strip())
    return [p for p in parts if p]


def by_paragraph(page: DocumentPage, max_size: int = 800) -> list[Chunk]:
    """Group paragraphs up to max_size, split oversized paragraphs, and retain metadata."""
    paragraphs = _split_paragraphs(page.text)
    if not paragraphs:
        return []

    expanded: list[str] = []
    for para in paragraphs:
        expanded.extend(_split_oversized(para, max_size=max_size))

    grouped: list[str] = []
    current = ""
    for para in expanded:
        if not current:
            current = para
        elif len(current) + 1 + len(para) <= max_size:
            current = f"{current} {para}"
        else:
            grouped.append(current)
            current = para
    if current:
        grouped.append(current)

    chunks: list[Chunk] = []
    for idx, text in enumerate(grouped):
        meta = dict(page.metadata)
        meta["chunk_index"] = idx
        chunks.append(
            Chunk(
                chunk_id=f"{page.document_id}:B:p{page.page:02d}:c{idx:03d}",
                document_id=page.document_id,
                source=page.source,
                page=page.page,
                text=text,
                metadata=meta,
            )
        )
    return chunks
