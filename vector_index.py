import numpy as np
from typing import Dict, List


class VectorIndex:
    """
    Dynamic vector index (Section A).

    Rules:
    - Dot-product similarity on L2-normalized vectors.
    - insert: succeeds iff ID does not exist.
    - delete: succeeds iff ID exists.
    - search: return shape (num_queries, min(k, n_active)); IDs sorted by descending dot product.
    - Each of insert/delete/search must be at most 20 physical lines.
    """

    def __init__(self, dim: int):
        self.dim = int(dim)
        self._store: Dict[int, np.ndarray] = {}
        self._pos: Dict[int, int] = {}
        self._ids = np.empty((0,), dtype=np.int64)
        self._vectors = np.empty((0, self.dim), dtype=np.float32)

        # Simple inverted-file-like structure.
        # Vectors are assigned to buckets based on the sign pattern
        # of their first few dimensions.
        self._bucket_bits = min(8, self.dim)
        self._buckets: Dict[int, List[int]] = {}

    def _bucket_key(self, vec: np.ndarray) -> int:
        """Create a small integer bucket key from the signs of the first dimensions."""
        bits = vec[:self._bucket_bits] > 0
        key = 0
        for i, b in enumerate(bits):
            key |= int(b) << i
        return key

    def insert(self, batch: Dict[int, np.ndarray]) -> Dict[str, List[int]]:
        """Return {"succeeded": [...], "failed": [...]} preserving input order per list."""
        succeeded: List[int] = []
        failed: List[int] = []
        new_ids: List[int] = []
        new_vecs: List[np.ndarray] = []
        base = len(self._ids)
        for vid, vec in batch.items():
            vid = int(vid)
            if vid in self._store:
                failed.append(vid); continue
            vec = np.asarray(vec, dtype=np.float32)
            self._store[vid] = vec; self._pos[vid] = base + len(new_ids)
            self._buckets.setdefault(self._bucket_key(vec), []).append(vid)
            new_ids.append(vid); new_vecs.append(vec); succeeded.append(vid)
        if new_ids:
            self._ids = np.concatenate((self._ids, np.asarray(new_ids, dtype=np.int64)))
            self._vectors = np.vstack((self._vectors, np.stack(new_vecs)))
        return {"succeeded": succeeded, "failed": failed}

    def delete(self, ids: np.ndarray) -> Dict[str, List[int]]:
        """Return {"succeeded": [...], "failed": [...]} preserving input order per list."""
        succeeded: List[int] = []
        failed: List[int] = []
        for vid in np.asarray(ids, dtype=np.int64):
            vid = int(vid)
            if vid not in self._store:
                failed.append(vid); continue
            key = self._bucket_key(self._store[vid])
            if key in self._buckets:
                self._buckets[key].remove(vid)
            pos = self._pos.pop(vid); last = len(self._ids) - 1; last_id = int(self._ids[last])
            if pos != last:
                self._ids[pos] = last_id; self._vectors[pos] = self._vectors[last]; self._pos[last_id] = pos
            self._ids = self._ids[:last]; self._vectors = self._vectors[:last]
            del self._store[vid]; succeeded.append(vid)
        return {"succeeded": succeeded, "failed": failed}

    #@profile
    def search(self, queries: np.ndarray, k: int) -> np.ndarray:
        """Return (num_queries, min(k, n_active)) int64 array of vector IDs."""
        queries = np.asarray(queries, dtype=np.float32)
        n_q, n = queries.shape[0], len(self._ids)
        if n == 0 or int(k) <= 0:
            return np.empty((n_q, 0), dtype=np.int64)
        k_eff = min(int(k), n)
        out = np.empty((n_q, k_eff), dtype=np.int64)
        for qi, q in enumerate(queries):
            cand_ids = self._buckets.get(self._bucket_key(q), [])
            if len(cand_ids) < k_eff:
                cand_pos = np.arange(n, dtype=np.int64)
            else:
                cand_pos = np.fromiter((self._pos[int(x)] for x in cand_ids), dtype=np.int64)
            scores = self._vectors[cand_pos] @ q
            kk = min(k_eff, len(cand_pos))
            topk = np.argpartition(scores, len(scores) - kk)[-kk:]
            order = np.argsort(-scores[topk])
            out[qi, :kk] = self._ids[cand_pos[topk[order]]]
        return out