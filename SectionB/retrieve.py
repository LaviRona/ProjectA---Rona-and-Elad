"""Query-time retrieval: hybrid dense (centroid) + lexical (BM25) fusion.

Each page is scored by (1) the dot product of the query with the page's mean
chunk vector (dense semantic match) and (2) BM25 over the page's full text
(exact keyword match). Per query we min-max normalize each score to [0, 1] and
fuse: score = (1 - FUSION_W)*dense + FUSION_W*lexical. Fusion beats either signal
alone on the public queries (see RESULTS.md).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from embed import embed_queries
from index import load_index
from lexical import BM25Index
from utils import ARTIFACTS_DIR, K_EVAL

# Weight on the lexical (BM25) signal in [0, 1]. Tuned on public queries; the
# 0.15–0.25 range is a flat optimum, so 0.2 is robust (not overfit).
FUSION_W = 0.2


def _minmax(row: np.ndarray) -> np.ndarray:
    lo, hi = row.min(), row.max()
    return (row - lo) / (hi - lo + 1e-9)


def search_batch(
    queries: List[str],
    *,
    top_k: int = K_EVAL,
    artifacts_dir: Optional[Path] = None,
) -> List[List[int]]:
    """Return ranked page_id lists (best first) for each query."""
    root = artifacts_dir or ARTIFACTS_DIR
    centroids, page_ids = load_index(root)
    bm25 = BM25Index(root)
    page_ids_arr = np.asarray(page_ids)

    query_vectors = embed_queries(queries)
    if query_vectors.size == 0:
        return [[] for _ in queries]

    dense = query_vectors @ centroids.T  # (num_queries, num_pages)
    ranked: List[List[int]] = []
    for i, q in enumerate(queries):
        d = _minmax(dense[i])
        l = _minmax(bm25.scores(q))
        fused = (1.0 - FUSION_W) * d + FUSION_W * l
        top = np.argpartition(-fused, range(min(top_k, fused.size)))[:top_k]
        top = top[np.argsort(-fused[top])]
        ranked.append([int(page_ids_arr[j]) for j in top])
    return ranked
