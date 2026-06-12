import numpy as np
from typing import Dict, List

try:
    profile
except NameError:
    profile = lambda f: f

class VectorIndex:
    """
    Dynamic vector index - optimized exact full scan.

    Uses:
    - Contiguous float32 matrix storage with capacity doubling.
    - ID-to-row dictionary for O(1) existence checks and deletion.
    - Tombstone deletion with periodic compaction.
    - Batched full dot-product search for exact Recall@10.
    - Exact top-k selection, optimized for small k.
    """

    def __init__(self, dim: int):
        self.dim = int(dim)
        self._cap = 0
        self._n = 0
        self._n_active = 0
        self._vectors = np.empty((0, self.dim), dtype=np.float32)
        self._ids = np.empty((0,), dtype=np.int64)
        self._active = np.empty((0,), dtype=bool)
        self._id2row: Dict[int, int] = {}

    def _grow(self, extra: int) -> None:
        need = self._n + extra
        if need <= self._cap:
            return
        new_cap = max(need, max(1, self._cap) * 2)
        vectors = np.empty((new_cap, self.dim), dtype=np.float32)
        ids = np.empty((new_cap,), dtype=np.int64)
        active = np.zeros((new_cap,), dtype=bool)
        if self._n:
            vectors[:self._n] = self._vectors[:self._n]
            ids[:self._n] = self._ids[:self._n]
            active[:self._n] = self._active[:self._n]
        self._vectors, self._ids, self._active, self._cap = vectors, ids, active, new_cap

    def _compact(self) -> None:
        live = np.flatnonzero(self._active[:self._n])
        m = live.size
        if m == self._n:
            return
        self._vectors[:m] = self._vectors[live]
        self._ids[:m] = self._ids[live]
        self._active[:m] = True
        self._active[m:self._n] = False
        self._n = m
        self._n_active = m
        self._id2row = {int(self._ids[i]): i for i in range(m)}

    def _maybe_compact(self) -> None:
        if self._n and self._n_active < 0.98 * self._n:
            self._compact()

    def insert(self, batch: Dict[int, np.ndarray]) -> Dict[str, List[int]]:
        """Return {"succeeded": [...], "failed": [...]} preserving input order per list."""
        succeeded: List[int] = []; failed: List[int] = []
        new_ids: List[int] = []; new_vecs: List[np.ndarray] = []
        for vid, vec in batch.items():
            vid = int(vid)
            if vid in self._id2row:
                failed.append(vid); continue
            new_ids.append(vid); new_vecs.append(np.asarray(vec, dtype=np.float32)); succeeded.append(vid)
        if new_ids:
            m, s = len(new_ids), self._n
            self._grow(m)
            self._vectors[s:s + m] = np.stack(new_vecs)
            self._ids[s:s + m] = np.asarray(new_ids, dtype=np.int64)
            self._active[s:s + m] = True
            for j, vid in enumerate(new_ids):
                self._id2row[vid] = s + j
            self._n += m; self._n_active += m
        return {"succeeded": succeeded, "failed": failed}

    def delete(self, ids: np.ndarray) -> Dict[str, List[int]]:
        """Return {"succeeded": [...], "failed": [...]} preserving input order per list."""
        succeeded: List[int] = []; failed: List[int] = []
        for vid in np.asarray(ids, dtype=np.int64):
            vid = int(vid); row = self._id2row.pop(vid, -1)
            if row < 0:
                failed.append(vid); continue
            if self._active[row]:
                self._active[row] = False; self._n_active -= 1
            succeeded.append(vid)
        self._maybe_compact()
        return {"succeeded": succeeded, "failed": failed}

    def _topk_small(self, scores: np.ndarray, k: int) -> np.ndarray:
        """Exact top-k by repeated argmax. Usually good for small k like 10."""
        out = np.empty((scores.shape[0], k), dtype=np.int64)
        rows = np.arange(scores.shape[0])
        for i in range(k):
            mx = np.argmax(scores, axis=1)
            out[:, i] = mx
            scores[rows, mx] = -np.inf
        return out

    def _topk_partition(self, scores: np.ndarray, k: int) -> np.ndarray:
        """Exact top-k by argpartition plus sorting selected values."""
        idx = np.argpartition(scores, scores.shape[1] - k, axis=1)[:, -k:]
        vals = np.take_along_axis(scores, idx, axis=1)
        order = np.argsort(-vals, axis=1)
        return np.take_along_axis(idx, order, axis=1)

    def _topk(self, scores: np.ndarray, k: int) -> np.ndarray:
        if k <= 16:
            return self._topk_small(scores, k)
        return self._topk_partition(scores, k)

    @profile
    def search(self, queries: np.ndarray, k: int) -> np.ndarray:
        """Return (num_queries, min(k, n_active)) int64 array of vector IDs."""
        q = np.asarray(queries, dtype=np.float32)
        nq, n = q.shape[0], self._n_active
        if n == 0 or int(k) <= 0:
            return np.empty((nq, 0), dtype=np.int64)
        keff = min(int(k), n)
        self._maybe_compact()
        scores = q @ self._vectors[:self._n].T
        if self._n_active != self._n:
            scores[:, ~self._active[:self._n]] = -np.inf
        return self._ids[self._topk(scores, keff)]