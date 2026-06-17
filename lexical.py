"""BM25 lexical index — built offline, loaded at query time for hybrid retrieval.

Dense MiniLM embeddings miss exact tokens (numbers, names, dates) that factoid
queries hinge on. A BM25 keyword score over the *full, untruncated* page text
complements the dense score; fusing the two (see retrieve.py) beats either alone.

Persisted as compact CSR-by-term arrays so query-time scoring touches only the
postings of the query's terms.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

BM25_PAGES_NAME = "bm25_pages.npy"      # CSR: concatenated page indices
BM25_TF_NAME = "bm25_tf.npy"            # CSR: matching term frequencies
BM25_OFFSETS_NAME = "bm25_offsets.npy"  # CSR: per-term slice offsets (len V+1)
BM25_STATS_NAME = "bm25_stats.json"     # vocab, idf, doc lengths, page_ids, params

TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def build_and_save(
    page_ids: List[int], texts: List[str], out_dir: Path, *, k1: float = 1.5, b: float = 0.75
) -> None:
    """Build a BM25 index over (page_ids, texts) and persist it to out_dir.

    `page_ids[i]` is the page for `texts[i]`; this order must match the dense
    index rows.
    """
    N = len(texts)
    doc_len = np.zeros(N, dtype=np.float32)
    df: Counter = Counter()
    term_id: Dict[str, int] = {}
    postings: Dict[int, List[tuple]] = defaultdict(list)  # term_id -> [(doc, tf)]

    for d, text in enumerate(texts):
        toks = tokenize(text)
        doc_len[d] = len(toks)
        for t, f in Counter(toks).items():
            tid = term_id.setdefault(t, len(term_id))
            postings[tid].append((d, f))
            df[t] += 1

    V = len(term_id)
    offsets = np.zeros(V + 1, dtype=np.int64)
    for tid in range(V):
        offsets[tid + 1] = offsets[tid] + len(postings[tid])
    pages = np.empty(int(offsets[-1]), dtype=np.int32)
    tfs = np.empty(int(offsets[-1]), dtype=np.float32)
    for tid in range(V):
        s = int(offsets[tid])
        for j, (d, f) in enumerate(postings[tid]):
            pages[s + j] = d
            tfs[s + j] = f

    avgdl = float(doc_len.mean()) if N else 0.0
    idf = {t: math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5)) for t in term_id}

    np.save(out_dir / BM25_PAGES_NAME, pages)
    np.save(out_dir / BM25_TF_NAME, tfs)
    np.save(out_dir / BM25_OFFSETS_NAME, offsets)
    (out_dir / BM25_STATS_NAME).write_text(
        json.dumps(
            {
                "term_id": term_id,
                "idf": idf,
                "doc_len": doc_len.tolist(),
                "page_ids": [int(p) for p in page_ids],
                "avgdl": avgdl,
                "k1": k1,
                "b": b,
            }
        ),
        encoding="utf-8",
    )


class BM25Index:
    """Loaded BM25 index; `scores(query)` returns one BM25 score per page."""

    def __init__(self, artifacts_dir: Path):
        self.pages = np.load(artifacts_dir / BM25_PAGES_NAME)
        self.tfs = np.load(artifacts_dir / BM25_TF_NAME)
        self.offsets = np.load(artifacts_dir / BM25_OFFSETS_NAME)
        stats = json.loads((artifacts_dir / BM25_STATS_NAME).read_text(encoding="utf-8"))
        self.term_id: Dict[str, int] = stats["term_id"]
        self.idf: Dict[str, float] = stats["idf"]
        self.doc_len = np.asarray(stats["doc_len"], dtype=np.float32)
        self.page_ids: List[int] = [int(p) for p in stats["page_ids"]]
        self.avgdl = float(stats["avgdl"])
        self.k1 = float(stats["k1"])
        self.b = float(stats["b"])
        self._denom_norm = self.k1 * (1 - self.b + self.b * self.doc_len / self.avgdl)

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.page_ids), dtype=np.float32)
        for term in set(tokenize(query)):
            tid = self.term_id.get(term)
            if tid is None:
                continue
            s, e = int(self.offsets[tid]), int(self.offsets[tid + 1])
            docs = self.pages[s:e]
            tf = self.tfs[s:e]
            contrib = self.idf[term] * (tf * (self.k1 + 1)) / (tf + self._denom_norm[docs])
            np.add.at(out, docs, contrib)
        return out
