"""Experiment A: sweep chunk size/overlap; score mean-pool (and a few variants).

For each (CHUNK_WORDS, OVERLAP_WORDS) config: re-chunk + re-embed the corpus,
build page centroids, and log mean-pool NDCG@10 on the 50 public queries to
RESULTS.md. A few extra aggregations are logged from the same chunk vectors for
the record. The best (config, mean-pool) centroids are cached for Experiment B.

Offline only (re-embeds the corpus per config; ~minutes each on GPU).

    cd SectionB && python scripts/sweep_chunks.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

STUDENT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(STUDENT_ROOT))

import chunk as chunk_mod
from chunk import chunk_corpus
from embed import embed_queries, embed_texts
from eval import mean_ndcg_at_k
from index import page_centroids
from utils import ARTIFACTS_DIR, K_EVAL, iter_entries, load_public_queries
from _results import log_result

CONFIGS = [(128, 24), (200, 40), (300, 50), (384, 64)]


def topk_pages(page_scores: np.ndarray, unique_pages: np.ndarray, k: int) -> list[list[int]]:
    out = []
    for row in page_scores:
        top = np.argpartition(-row, range(min(k, row.size)))[:k]
        top = top[np.argsort(-row[top])]
        out.append([int(unique_pages[i]) for i in top])
    return out


def main() -> None:
    records = list(iter_entries())
    rows = load_public_queries()
    queries = [r["query"] for r in rows]
    gt = [set(r["relevant_page_ids"]) for r in rows]
    qv = embed_queries(queries).astype(np.float32)

    best = {"ndcg": -1.0}
    for cw, ov in CONFIGS:
        chunk_mod.CHUNK_WORDS = cw
        chunk_mod.OVERLAP_WORDS = ov
        chunks = chunk_corpus(records)
        print(f"\n=== chunk {cw}w/{ov}o: {len(chunks):,} chunks; embedding... ===")
        V = embed_texts([c.text for c in chunks], progress=True).astype(np.float32)
        pid = np.asarray([c.page_id for c in chunks])
        cid = np.asarray([c.chunk_id for c in chunks])
        unique, inv = np.unique(pid, return_inverse=True)
        P = unique.size
        counts = np.bincount(inv, minlength=P).astype(np.float32)

        scores = qv @ V.T  # (nq, nchunks)
        cfg = f"{cw}w/{ov}o ({len(chunks)//1000}k chunks)"

        # mean-pool == dot with centroid
        centroids, cpages = page_centroids(V, pid.tolist())
        mean_scores = qv @ centroids.T
        ndcg_mean = mean_ndcg_at_k(topk_pages(mean_scores, np.asarray(cpages), K_EVAL), gt, k=K_EVAL)
        log_result("chunkA:mean", cfg, ndcg_mean, notes="mean of chunk vectors (centroid)")

        # max-pool (reference)
        mx = np.full((qv.shape[0], P), -1e9, dtype=np.float32)
        for i in range(qv.shape[0]):
            np.maximum.at(mx[i], inv, scores[i])
        log_result("chunkA:max", cfg, mean_ndcg_at_k(topk_pages(mx, unique, K_EVAL), gt, k=K_EVAL), notes="reference")

        # mean + 0.5*max combined
        combo = mean_scores + 0.5 * mx
        log_result("chunkA:mean+.5max", cfg, mean_ndcg_at_k(topk_pages(combo, np.asarray(cpages), K_EVAL), gt, k=K_EVAL), notes="combine")

        # lead-only (chunk0)
        m0 = (cid == 0)
        c0 = np.full((qv.shape[0], P), -1e9, dtype=np.float32)
        for i in range(qv.shape[0]):
            np.maximum.at(c0[i], inv[m0], scores[i][m0])
        log_result("chunkA:chunk0", cfg, mean_ndcg_at_k(topk_pages(c0, unique, K_EVAL), gt, k=K_EVAL), notes="lead-only ~baseline")

        if ndcg_mean > best["ndcg"]:
            best = {"ndcg": ndcg_mean, "cw": cw, "ov": ov, "centroids": centroids, "pages": cpages}

    np.savez(
        ARTIFACTS_DIR / "_best_centroids.npz",
        centroids=best["centroids"].astype(np.float32),
        page_ids=np.asarray(best["pages"]),
        cw=best["cw"], ov=best["ov"],
    )
    print(f"\nBEST mean-pool: {best['cw']}w/{best['ov']}o NDCG@10={best['ndcg']:.4f} (cached)")


if __name__ == "__main__":
    main()
