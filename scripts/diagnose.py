"""Offline diagnostic: compare page-aggregation strategies on the 50 public queries.

Loads the *already-built* artifacts (no corpus re-embedding) and the public
queries, embeds the queries once, then prints mean NDCG@10 for several ways of
collapsing per-chunk scores into a per-page score. Use this to decide which
aggregation `retrieve.py` should use.

Run:
    cd SectionB
    python scripts/diagnose.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

STUDENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDENT_ROOT))

from chunk import chunk_corpus
from embed import embed_queries, embed_texts
from eval import mean_ndcg_at_k
from utils import K_EVAL, iter_entries, load_public_queries


def rank_pages(page_scores: np.ndarray, unique_pages: np.ndarray, k: int) -> list[list[int]]:
    """Top-k page_ids per query from a (num_queries, num_pages) score matrix."""
    ranked = []
    for row in page_scores:
        top = np.argpartition(-row, range(min(k, row.size)))[:k]
        top = top[np.argsort(-row[top])]
        ranked.append([int(unique_pages[i]) for i in top])
    return ranked


def main() -> None:
    # Self-contained: chunk + embed the corpus in memory so the comparison does
    # not depend on whatever level the artifacts were last built at. This embeds
    # the full corpus (~minutes on GPU); it is an offline experiment, not run().
    chunks = chunk_corpus(list(iter_entries()))
    print(f"Embedding {len(chunks):,} chunks for diagnosis...")
    vectors = embed_texts([c.text for c in chunks], progress=True)
    page_ids = np.asarray([c.page_id for c in chunks])
    chunk_ids = np.asarray([c.chunk_id for c in chunks])

    # Map each chunk to a contiguous page index 0..P-1.
    unique_pages, page_index = np.unique(page_ids, return_inverse=True)
    P = unique_pages.size
    counts = np.bincount(page_index, minlength=P)

    rows = load_public_queries()
    queries = [r["query"] for r in rows]
    ground_truth = [set(r["relevant_page_ids"]) for r in rows]

    qv = embed_queries(queries).astype(np.float32)
    scores = qv @ vectors.T  # (num_queries, num_chunks)
    NEG = -1e9

    def evaluate(name: str, page_scores: np.ndarray) -> None:
        ranked = rank_pages(page_scores, unique_pages, K_EVAL)
        ndcg = mean_ndcg_at_k(ranked, ground_truth, k=K_EVAL)
        print(f"  {name:<22} NDCG@10 = {ndcg:.4f}")

    def max_per_page(sc: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        ps = np.full((sc.shape[0], P), NEG, dtype=np.float32)
        for i in range(sc.shape[0]):
            row = sc[i]
            idx = page_index
            if mask is not None:
                row = row[mask]
                idx = page_index[mask]
            np.maximum.at(ps[i], idx, row)
        return ps

    def mean_per_page(sc: np.ndarray) -> np.ndarray:
        ps = np.zeros((sc.shape[0], P), dtype=np.float32)
        for i in range(sc.shape[0]):
            np.add.at(ps[i], page_index, sc[i])
        return ps / np.maximum(counts, 1)

    def topk_mean_per_page(sc: np.ndarray, k: int) -> np.ndarray:
        ps = np.full((sc.shape[0], P), NEG, dtype=np.float32)
        for i in range(sc.shape[0]):
            # sort chunks by (page asc, score desc); keep within-page rank < k
            order = np.lexsort((-sc[i], page_index))
            sp = page_index[order]
            ss = sc[i][order]
            start = np.searchsorted(sp, np.arange(P))      # first row of each page
            within = np.arange(sp.size) - start[sp]         # rank within its page
            keep = within < k
            acc = np.zeros(P, dtype=np.float32)
            np.add.at(acc, sp[keep], ss[keep])
            denom = np.minimum(counts, k)
            ps[i] = acc / np.maximum(denom, 1)
        return ps

    print(f"queries={len(queries)} pages={P} chunks={vectors.shape[0]}")
    print("Baselines / sanity:")
    evaluate("chunk0_max (~baseline)", max_per_page(scores, mask=(chunk_ids == 0)))
    evaluate("all_max (current)", max_per_page(scores))
    print("Alternatives:")
    evaluate("mean", mean_per_page(scores))
    for k in (2, 3, 5):
        evaluate(f"topk_mean k={k}", topk_mean_per_page(scores, k))
    for K in (2, 3, 4):
        evaluate(f"firstK_max K={K}", max_per_page(scores, mask=(chunk_ids < K)))
    # length-penalized max: max - alpha*log(n_chunks)
    base_max = max_per_page(scores)
    for alpha in (0.05, 0.1, 0.15, 0.2, 0.3):
        evaluate(f"len_pen_max a={alpha}", base_max - alpha * np.log(counts)[None, :])

    # normalized centroid: cosine of query with the (re-normalized) mean chunk vector
    def normalized_centroid_scores() -> np.ndarray:
        centroids = np.zeros((P, vectors.shape[1]), dtype=np.float32)
        np.add.at(centroids, page_index, vectors)
        centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-9)
        return qv @ centroids.T
    print("More:")
    evaluate("norm_centroid", normalized_centroid_scores())

    # log-sum-exp pooling: interpolates max (T->0) and mean (T large)
    def lse_per_page(sc: np.ndarray, T: float) -> np.ndarray:
        ps = np.full((sc.shape[0], P), 0.0, dtype=np.float64)
        for i in range(sc.shape[0]):
            mx = np.full(P, NEG, dtype=np.float64)
            np.maximum.at(mx, page_index, sc[i])
            acc = np.zeros(P, dtype=np.float64)
            np.add.at(acc, page_index, np.exp((sc[i] - mx[page_index]) / T))
            ps[i] = mx + T * np.log(acc)
        return ps
    for T in (0.05, 0.1, 0.2):
        evaluate(f"logsumexp T={T}", lse_per_page(scores, T))


if __name__ == "__main__":
    main()
