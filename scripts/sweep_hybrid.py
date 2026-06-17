"""Experiment B: hybrid dense (centroid) + lexical (BM25) retrieval.

Many queries are factoid (exact numbers, names, dates) where dense MiniLM is
weak. Build a pure-Python BM25 over full (untruncated) page text, fuse with the
dense centroid score, and sweep the fusion weight on the 50 public queries.

Reuses the best centroids cached by sweep_chunks.py (artifacts/_best_centroids.npz).
Offline only.

    cd SectionB && python scripts/sweep_hybrid.py
"""
from __future__ import annotations

import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

STUDENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDENT_ROOT))

from embed import embed_queries
from eval import mean_ndcg_at_k
from utils import ARTIFACTS_DIR, K_EVAL, entry_text, iter_entries, load_public_queries
from _results import log_result

TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, docs_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.N = len(docs_tokens)
        self.dl = np.array([len(d) for d in docs_tokens], dtype=np.float32)
        self.avgdl = float(self.dl.mean()) if self.N else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        df: Counter = Counter()
        for i, toks in enumerate(docs_tokens):
            tf = Counter(toks)
            for t, f in tf.items():
                self.postings[t].append((i, f))
                df[t] += 1
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(self.N, dtype=np.float32)
        for t in set(tokenize(query)):
            if t not in self.postings:
                continue
            idf = self.idf[t]
            for i, f in self.postings[t]:
                denom = f + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl)
                out[i] += idf * (f * (self.k1 + 1)) / denom
        return out


def minmax(row: np.ndarray) -> np.ndarray:
    lo, hi = row.min(), row.max()
    return (row - lo) / (hi - lo + 1e-9)


def topk_pages(page_scores: np.ndarray, page_ids: np.ndarray, k: int) -> list[list[int]]:
    out = []
    for row in page_scores:
        top = np.argpartition(-row, range(min(k, row.size)))[:k]
        top = top[np.argsort(-row[top])]
        out.append([int(page_ids[i]) for i in top])
    return out


def main() -> None:
    cache = np.load(ARTIFACTS_DIR / "_best_centroids.npz")
    centroids = cache["centroids"].astype(np.float32)
    page_ids = cache["page_ids"]
    cw, ov = int(cache["cw"]), int(cache["ov"])
    cfg = f"{cw}w/{ov}o + BM25"

    # Full-text per page, ordered to match centroid rows.
    text_by_pid = {int(r["page_id"]): entry_text(r) for r in iter_entries()}
    docs = [tokenize(text_by_pid.get(int(p), "")) for p in page_ids]
    print(f"Building BM25 over {len(docs):,} pages...")
    bm25 = BM25(docs)

    rows = load_public_queries()
    queries = [r["query"] for r in rows]
    gt = [set(r["relevant_page_ids"]) for r in rows]
    qv = embed_queries(queries).astype(np.float32)
    dense = qv @ centroids.T                                  # (nq, P)
    lexical = np.vstack([bm25.scores(q) for q in queries])    # (nq, P)

    dn = np.vstack([minmax(r) for r in dense])
    ln = np.vstack([minmax(r) for r in lexical])

    # dense-only and lexical-only references
    log_result("hybrid:dense", cfg, mean_ndcg_at_k(topk_pages(dense, page_ids, K_EVAL), gt, k=K_EVAL), notes="w=0")
    log_result("hybrid:lexical", cfg, mean_ndcg_at_k(topk_pages(lexical, page_ids, K_EVAL), gt, k=K_EVAL), notes="BM25 only")

    for w in (0.1, 0.15, 0.2, 0.25, 0.3):
        fused = (1 - w) * dn + w * ln
        ndcg = mean_ndcg_at_k(topk_pages(fused, page_ids, K_EVAL), gt, k=K_EVAL)
        log_result("hybrid:wsum", f"{cfg} w={w}", ndcg, notes="minmax weighted sum")

    # Reciprocal rank fusion (rank-based, scale-free)
    def rrf(scores: np.ndarray, k: int = 60) -> np.ndarray:
        ranks = np.argsort(np.argsort(-scores, axis=1), axis=1)  # 0 = best
        return 1.0 / (k + ranks + 1)
    fused_rrf = rrf(dense) + rrf(lexical)
    log_result("hybrid:rrf", cfg, mean_ndcg_at_k(topk_pages(fused_rrf, page_ids, K_EVAL), gt, k=K_EVAL), notes="reciprocal rank fusion")


if __name__ == "__main__":
    main()
