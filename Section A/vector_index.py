import numpy as np
from typing import Dict, List


class VectorIndex:
    """
    Dynamic vector index (Section A) - exact flat scan.

    Storage is one contiguous, capacity-doubling float32 matrix (no per-search
    re-stacking, unlike the naive baseline). Deletes are tombstoned and the
    array is compacted when bloated. Search is a single batched matmul
    `queries @ vectors.T` followed by an exact top-k via k argmax passes
    (cheaper than a full argpartition for small k). Recall is always 1.0.
    Similarity is the dot product on already-L2-normalized vectors.
    """

    def __init__(self, dim: int):
        self.dim = int(dim)
        self._cap = 0
        self._n = 0            # rows used (incl. tombstones)
        self._n_active = 0     # live rows
        self._vectors = np.empty((0, self.dim), dtype=np.float32)
        self._ids = np.empty((0,), dtype=np.int64)
        self._active = np.empty((0,), dtype=bool)
        self._id2row: Dict[int, int] = {}

    def _grow(self, extra: int) -> None:
        need = self._n + extra
        if need <= self._cap:
            return
        new_cap = max(need, max(1, self._cap) * 2)
        v = np.empty((new_cap, self.dim), dtype=np.float32)
        v[:self._n] = self._vectors[:self._n]
        i = np.empty((new_cap,), dtype=np.int64)
        i[:self._n] = self._ids[:self._n]
        a = np.zeros((new_cap,), dtype=bool)
        a[:self._n] = self._active[:self._n]
        self._vectors, self._ids, self._active, self._cap = v, i, a, new_cap

    def insert(self, batch: Dict[int, np.ndarray]) -> Dict[str, List[int]]:
        """Return {"succeeded": [...], "failed": [...]} preserving input order per list."""
        succeeded: List[int] = []
        failed: List[int] = []
        new_ids: List[int] = []
        new_vecs: List[np.ndarray] = []
        seen = set()
        for vid, vec in batch.items():
            vid = int(vid)
            if vid in self._id2row or vid in seen:
                failed.append(vid)
                continue
            seen.add(vid)
            new_ids.append(vid)
            new_vecs.append(np.asarray(vec, dtype=np.float32))
            succeeded.append(vid)
        if new_ids:
            m = len(new_ids)
            self._grow(m)
            s = self._n
            self._vectors[s:s + m] = np.stack(new_vecs)
            self._ids[s:s + m] = np.asarray(new_ids, dtype=np.int64)
            self._active[s:s + m] = True
            for j in range(m):
                self._id2row[new_ids[j]] = s + j
            self._n += m
            self._n_active += m
        return {"succeeded": succeeded, "failed": failed}

    def delete(self, ids: np.ndarray) -> Dict[str, List[int]]:
        """Return {"succeeded": [...], "failed": [...]} preserving input order per list."""
        succeeded: List[int] = []
        failed: List[int] = []
        for vid in np.asarray(ids, dtype=np.int64):
            vid = int(vid)
            row = self._id2row.pop(vid, -1)
            if row < 0:
                failed.append(vid)
                continue
            if self._active[row]:
                self._active[row] = False
                self._n_active -= 1
            succeeded.append(vid)
        self._maybe_compact()
        return {"succeeded": succeeded, "failed": failed}

    def _maybe_compact(self) -> None:
        if self._n == 0 or self._n_active >= 0.85 * self._n:
            return
        live = np.flatnonzero(self._active[:self._n])
        m = live.size
        self._vectors[:m] = self._vectors[live]
        self._ids[:m] = self._ids[live]
        self._active[:m] = True
        if m < self._n:
            self._active[m:self._n] = False
        self._n = m
        self._n_active = m
        self._id2row = {int(self._ids[r]): r for r in range(m)}

    def _topk(self, S: np.ndarray, k: int) -> np.ndarray:
        """Exact indices of the k largest per row, descending, via k argmax passes."""
        out = np.empty((S.shape[0], k), dtype=np.int64)
        ar = np.arange(S.shape[0])
        for i in range(k):
            mx = np.argmax(S, axis=1)
            out[:, i] = mx
            S[ar, mx] = -np.inf
        return out

    def search(self, queries: np.ndarray, k: int) -> np.ndarray:
        """Return (num_queries, min(k, n_active)) int64 array of vector IDs."""
        q = np.asarray(queries, dtype=np.float32)
        nq, n = q.shape[0], self._n_active
        if n == 0 or int(k) <= 0:
            return np.empty((nq, 0), dtype=np.int64)
        keff = min(int(k), n)
        S = q @ self._vectors[:self._n].T
        if self._n_active != self._n:                 # mask tombstoned columns
            S[:, ~self._active[:self._n]] = -np.inf
        return self._ids[self._topk(S, keff)]
