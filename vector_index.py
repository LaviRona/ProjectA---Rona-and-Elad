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
                failed.append(vid)
            else:
                vec = np.asarray(vec, dtype=np.float32)
                self._store[vid] = vec; self._pos[vid] = base + len(new_ids)
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
            pos = self._pos.pop(vid); last = len(self._ids) - 1; last_id = int(self._ids[last])
            if pos != last:
                self._ids[pos] = last_id; self._vectors[pos] = self._vectors[last]; self._pos[last_id] = pos
            self._ids = self._ids[:last]; self._vectors = self._vectors[:last]
            del self._store[vid]; succeeded.append(vid)
        return {"succeeded": succeeded, "failed": failed}

    def search(self, queries: np.ndarray, k: int) -> np.ndarray:
        """Return (num_queries, min(k, n_active)) int64 array of vector IDs."""
        queries = np.asarray(queries, dtype=np.float32)
        n_q = queries.shape[0]
        if len(self._ids) == 0 or int(k) <= 0:
            return np.empty((n_q, 0), dtype=np.int64)
        k_eff = min(int(k), len(self._ids))
        scores = queries @ self._vectors.T
        topk = np.argpartition(-scores, kth=k_eff - 1, axis=1)[:, :k_eff]
        vals = np.take_along_axis(scores, topk, axis=1)
        order = np.argsort(-vals, axis=1)
        return self._ids[np.take_along_axis(topk, order, axis=1)]