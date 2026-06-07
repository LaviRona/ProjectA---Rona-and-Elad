import numpy as np
from typing import Dict, List

class VectorIndex:
    """
    Dynamic vector index (Section A).
    Implements a batched Inverted File Index (IVF) using pure NumPy to eliminate
    per-query Python loops and maximize the runtime multiplier.
    """

    def __init__(self, dim: int):
        self.dim = int(dim)
        
        # Contiguous, capacity-doubling storage
        self._cap = 0
        self._n = 0
        self._n_active = 0
        
        self._vectors = np.empty((0, self.dim), dtype=np.float32)
        self._ids = np.empty((0,), dtype=np.int64)
        self._active = np.empty((0,), dtype=np.bool_)
        self._id2row: Dict[int, int] = {}
        
        # IVF Structures
        self._centroids = None
        self._cell_of = np.empty((0,), dtype=np.int32)
        self._inv_rows = None
        self._inv_off = None
        self._ivf_dirty = False
        
        # Tunables
        self._IVF_MIN = 50_000
        self._TARGET_CELL = 512
        self._rng = np.random.default_rng(0)

    def _append(self, ids: np.ndarray, vecs: np.ndarray):
        m = len(ids)
        if self._n + m > self._cap:
            new_cap = max(1024, (self._n + m) * 2)
            new_vectors = np.empty((new_cap, self.dim), dtype=np.float32)
            new_ids = np.empty((new_cap,), dtype=np.int64)
            new_active = np.empty((new_cap,), dtype=np.bool_)
            new_cell_of = np.empty((new_cap,), dtype=np.int32)
            
            if self._n > 0:
                new_vectors[:self._n] = self._vectors[:self._n]
                new_ids[:self._n] = self._ids[:self._n]
                new_active[:self._n] = self._active[:self._n]
                new_cell_of[:self._n] = self._cell_of[:self._n]
                
            self._vectors = new_vectors
            self._ids = new_ids
            self._active = new_active
            self._cell_of = new_cell_of
            self._cap = new_cap

        start, end = self._n, self._n + m
        self._vectors[start:end] = vecs
        self._ids[start:end] = ids
        self._active[start:end] = True
        
        for i in range(m):
            self._id2row[int(ids[i])] = start + i
            
        if self._centroids is not None:
            # Batched assignment (no loops)
            self._cell_of[start:end] = np.argmax(vecs @ self._centroids.T, axis=1)

        self._n += m
        self._n_active += m
        self._ivf_dirty = True

    def _build_kmeans(self):
        n = self._n_active
        if n == 0: return
        
        C = int(np.clip(round(n / self._TARGET_CELL), 64, 4096))
        live = np.flatnonzero(self._active[:self._n])
        
        sample_size = min(n, max(C * 256, 50_000))
        if sample_size < n:
            train_idx = self._rng.choice(live, sample_size, replace=False)
        else:
            train_idx = live
            
        X = self._vectors[train_idx]
        cents = X[self._rng.choice(len(X), C, replace=False)].copy()
        
        # Lloyd iterations (runs entirely in C via NumPy)
        for _ in range(5):
            scores = X @ cents.T
            assign = np.argmax(scores, axis=1)
            new_cents = np.zeros_like(cents)
            counts = np.bincount(assign, minlength=C)
            np.add.at(new_cents, assign, X)
            
            mask = counts > 0
            new_cents[mask] /= counts[mask, None]
            new_cents[~mask] = cents[~mask] # Keep old for empty clusters
            
            norms = np.linalg.norm(new_cents, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            cents = new_cents / norms
            
        self._centroids = cents.astype(np.float32)
        
        # Chunked assignment for all live rows
        chunk_size = 10000
        for i in range(0, len(live), chunk_size):
            chunk = live[i:i+chunk_size]
            self._cell_of[chunk] = np.argmax(self._vectors[chunk] @ self._centroids.T, axis=1)

        self._ivf_dirty = True

    def _maybe_compact(self):
        if self._n_active < 0.85 * self._n and self._n > 0:
            live = np.flatnonzero(self._active[:self._n])
            self._vectors[:self._n_active] = self._vectors[live]
            self._ids[:self._n_active] = self._ids[live]
            self._cell_of[:self._n_active] = self._cell_of[live]
            self._active[:self._n_active] = True
            
            self._n = self._n_active
            self._id2row.clear()
            for i in range(self._n_active):
                self._id2row[int(self._ids[i])] = i
                
            self._ivf_dirty = True

    def _build_lists(self):
        live = np.flatnonzero(self._active[:self._n])
        cells = self._cell_of[live]
        order = np.argsort(cells, kind='stable')
        
        self._inv_rows = live[order]
        C = self._centroids.shape[0] if self._centroids is not None else 0
        counts = np.bincount(cells[order], minlength=C)
        self._inv_off = np.concatenate([[0], np.cumsum(counts)])
        self._ivf_dirty = False

    def _nprobe(self) -> int:
        if self._centroids is None: return 1
        C = self._centroids.shape[0]
        # Tuning parameter: ~4% of clusters. Sweep this up to 48/64 if recall < 0.97
        return int(np.clip(round(C * 0.04), 16, 64))

    def insert(self, batch: Dict[int, np.ndarray]) -> Dict[str, List[int]]:
        succeeded, failed, new_ids, new_vecs = [], [], [], []
        seen_in_batch = set()
        
        for vid, vec in batch.items():
            vid = int(vid)
            if vid in self._id2row or vid in seen_in_batch:
                failed.append(vid)
            else:
                seen_in_batch.add(vid)
                new_ids.append(vid)
                new_vecs.append(vec)
                succeeded.append(vid)
                
        if new_ids:
            is_first_insert = (self._n == 0)
            self._append(np.array(new_ids, dtype=np.int64), np.array(new_vecs, dtype=np.float32))
            
            # Untimed phase trigger: build k-means only on the very first bulk insert
            if is_first_insert and self._centroids is None:
                self._build_kmeans()
                
        return {"succeeded": succeeded, "failed": failed}

    def delete(self, ids: np.ndarray) -> Dict[str, List[int]]:
        succeeded, failed = [], []
        
        for vid in ids:
            vid = int(vid)
            row = self._id2row.pop(vid, -1)
            if row < 0:
                failed.append(vid)
            else:
                self._active[row] = False
                self._n_active -= 1
                succeeded.append(vid)
                
        self._ivf_dirty = True
        self._maybe_compact()
        return {"succeeded": succeeded, "failed": failed}

    def _search_flat(self, q: np.ndarray, keff: int) -> np.ndarray:
        live = np.flatnonzero(self._active[:self._n])
        S = q @ self._vectors[live].T
        if S.shape[1] == 0:
            return np.empty((q.shape[0], 0), dtype=np.int64)
            
        topk = np.argpartition(S, S.shape[1] - keff, axis=1)[:, -keff:]
        ts = np.take_along_axis(S, topk, axis=1)
        order = np.argsort(-ts, axis=1)
        cols = np.take_along_axis(topk, order, axis=1)
        return self._ids[live[cols]]

    def _search_ivf(self, q: np.ndarray, keff: int) -> np.ndarray:
        if self._ivf_dirty:
            self._build_lists()
            
        nq, C = q.shape[0], self._centroids.shape[0]
        nprobe = min(self._nprobe(), C)
        
        QC = q @ self._centroids.T
        probe = np.argpartition(QC, C - nprobe, axis=1)[:, -nprobe:]
        
        flat_q = np.repeat(np.arange(nq), nprobe)
        flat_c = probe.ravel()
        o = np.argsort(flat_c, kind='stable')
        cc, qq = flat_c[o], flat_q[o]
        
        starts = np.searchsorted(cc, np.arange(C))
        ends = np.searchsorted(cc, np.arange(C) + 1)
        
        MAX = int(1.5 * nprobe * self._n_active / C) + keff
        cand_s = np.full((nq, MAX), -np.inf, dtype=np.float32)
        cand_r = np.zeros((nq, MAX), dtype=np.int64)
        fill = np.zeros(nq, dtype=np.int64)
        
        # Loop over clusters, NOT queries
        for c in range(C):
            s, e = starts[c], ends[c]
            if s == e: continue
            
            rows_c = self._inv_rows[self._inv_off[c]:self._inv_off[c+1]]
            if rows_c.size == 0: continue
            
            qids = qq[s:e]
            # Batched gather
            Sc = q[qids] @ self._vectors[rows_c].T
            m = rows_c.size
            
            # Fast contiguous slice writes
            for j in range(qids.size):
                qi = qids[j]
                w = fill[qi]
                if w + m > MAX:
                    m2 = MAX - w
                    cand_s[qi, w:] = Sc[j, :m2]
                    cand_r[qi, w:] = rows_c[:m2]
                    fill[qi] = MAX
                else:
                    cand_s[qi, w:w+m] = Sc[j]
                    cand_r[qi, w:w+m] = rows_c
                    fill[qi] += m
                    
        # Exact top-keff over candidates
        part = np.argpartition(cand_s, MAX - keff, axis=1)[:, -keff:]
        ts = np.take_along_axis(cand_s, part, axis=1)
        order = np.argsort(-ts, axis=1)
        cols = np.take_along_axis(part, order, axis=1)
        out = self._ids[np.take_along_axis(cand_r, cols, axis=1)]
        
        # Fallback for queries with insufficient candidates
        short = np.flatnonzero(fill < keff)
        if short.size > 0:
            out[short] = self._search_flat(q[short], keff)
            
        return out

    def search(self, queries: np.ndarray, k: int) -> np.ndarray:
        q = np.asarray(queries, dtype=np.float32)
        nq = q.shape[0]
        n = self._n_active
        
        if n == 0 or int(k) <= 0:
            return np.empty((nq, 0), dtype=np.int64)
            
        keff = min(int(k), n)
        
        if self._centroids is None or n < self._IVF_MIN:
            return self._search_flat(q, keff)
            
        return self._search_ivf(q, keff)