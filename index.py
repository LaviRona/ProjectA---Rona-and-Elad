"""Offline index build and load (not timed at grading)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from chunk import Chunk, chunk_corpus
from embed import embed_texts
import lexical
from utils import ARTIFACTS_DIR, ensure_artifacts_dir, entry_text, iter_entries

INDEX_VECTORS_NAME = "index_vectors.npy"
INDEX_META_NAME = "index_meta.json"


def page_centroids(
    chunk_vectors: np.ndarray, page_ids: List[int]
) -> Tuple[np.ndarray, List[int]]:
    """Average a page's chunk vectors into a single (un-normalized) page vector.

    The dot product of a query with this mean vector equals the *mean* of the
    query's similarities to the page's chunks — i.e. mean-pooling. This beats
    both whole-page embedding (truncated at 256 tokens) and max-pooling (which
    is length-biased: long pages get many chances to spuriously match). Keep the
    centroid un-normalized; re-normalizing it discards the topical-concentration
    signal and measured worse.
    """
    pid = np.asarray(page_ids)
    unique, inv = np.unique(pid, return_inverse=True)
    centroids = np.zeros((unique.size, chunk_vectors.shape[1]), dtype=np.float32)
    np.add.at(centroids, inv, chunk_vectors.astype(np.float32))
    counts = np.bincount(inv, minlength=unique.size).astype(np.float32)
    centroids /= counts[:, None]
    return centroids, [int(x) for x in unique]


def build_index(
    *,
    entries_dir: Optional[Path] = None,
    artifacts_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, List[int]]:
    """
    Embed the full corpus (chunked) and persist one mean vector per page.

    Returns (vectors, page_ids) where row i is the centroid for page_ids[i].
    """
    out_dir = artifacts_dir or ensure_artifacts_dir()
    records = list(iter_entries(entries_dir))
    chunks: List[Chunk] = chunk_corpus(records)
    texts = [c.text for c in chunks]
    print(f"Embedding {len(texts):,} chunks from {len(records):,} pages...")
    # progress=True shows a tqdm bar (elapsed / ETA / items per second).
    chunk_vectors = embed_texts(texts, progress=True)
    centroids, page_ids = page_centroids(chunk_vectors, [c.page_id for c in chunks])

    # One row per page; store as float16 to keep the artifact small (~20 MB).
    np.save(out_dir / INDEX_VECTORS_NAME, centroids.astype(np.float16))
    meta = {
        "page_ids": page_ids,
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "num_vectors": len(page_ids),
        "level": "page_centroid (mean of chunk vectors)",
    }
    (out_dir / INDEX_META_NAME).write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    # Lexical (BM25) index over full page text, in the SAME page order, for the
    # hybrid dense+keyword fusion done in retrieve.py.
    print("Building BM25 lexical index...")
    text_by_pid = {int(r["page_id"]): entry_text(r) for r in records}
    lexical.build_and_save(page_ids, [text_by_pid[p] for p in page_ids], out_dir)
    return centroids, page_ids


def load_index(
    artifacts_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, List[int]]:
    """Load precomputed vectors and page_id map from artifacts/."""
    root = artifacts_dir or ARTIFACTS_DIR
    # Cast back to float32 for fast, accurate dot-product search.
    vectors = np.load(root / INDEX_VECTORS_NAME).astype(np.float32)
    meta = json.loads((root / INDEX_META_NAME).read_text(encoding="utf-8"))
    page_ids = [int(x) for x in meta["page_ids"]]
    return vectors, page_ids
