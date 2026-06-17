"""Preprocessing and chunking.

The corpus articles are long (median ~1.2k words) while all-MiniLM-L6-v2
truncates at ~256 word-pieces. Embedding a whole page therefore only captures
its opening and discards the rest. We instead split each page into overlapping
fixed-size word windows so passages deep in an article become retrievable; at
query time each page is represented by its best-matching window.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

# Window sizing in *words*. all-MiniLM-L6-v2 truncates input at 256 word-pieces,
# so a chunk must stay under that or its tail is silently dropped. 180 words is
# ~235 word-pieces (incl. the prepended title) — the whole chunk is embedded, no
# truncation. The stride (150 < 180) makes consecutive windows overlap, so no
# region of a page is ever skipped: every word feeds the page centroid.
CHUNK_WORDS = 180
OVERLAP_WORDS = 30
# Bound runaway pages (a few articles exceed 30k words) so they don't dominate
# the index or the build time.
MAX_CHUNKS_PER_PAGE = 60


@dataclass
class Chunk:
    page_id: int
    chunk_id: int
    text: str


def chunk_entry(record: Dict[str, Any]) -> List[Chunk]:
    """
    Split one corpus entry into overlapping word-window retrieval units.

    The page title is prepended to every chunk so each passage carries entity
    context (e.g. the team/person/place named in the query).
    """
    page_id = int(record["page_id"])
    title = str(record.get("title", "")).strip()
    content = str(record.get("content", "")).strip()
    prefix = f"{title}\n\n" if title else ""

    words = content.split()
    if not words:
        # Title-only page: still index it as a single chunk.
        return [Chunk(page_id=page_id, chunk_id=0, text=prefix.strip() or str(page_id))]

    stride = max(1, CHUNK_WORDS - OVERLAP_WORDS)
    chunks: List[Chunk] = []
    for start in range(0, len(words), stride):
        window = " ".join(words[start : start + CHUNK_WORDS])
        chunks.append(
            Chunk(page_id=page_id, chunk_id=len(chunks), text=f"{prefix}{window}")
        )
        if start + CHUNK_WORDS >= len(words) or len(chunks) >= MAX_CHUNKS_PER_PAGE:
            break
    return chunks


def chunk_corpus(records: List[Dict[str, Any]]) -> List[Chunk]:
    chunks: List[Chunk] = []
    for record in records:
        chunks.extend(chunk_entry(record))
    return chunks
