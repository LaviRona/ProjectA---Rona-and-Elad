# Section B — Experiment results log (mean NDCG@10 on 50 public queries)

Append-only. Higher NDCG@10 is better. Baseline to beat: **0.2241** (original handout).

| timestamp | experiment | config | NDCG@10 | query_s | notes |
|-----------|------------|--------|---------|---------|-------|
| 2026-06-07 | baseline | whole-page embedding (MiniLM, truncated @256 tok) | 0.2241 | 3.60 | original handout |
| 2026-06-07 | chunk+maxpool | 200w/40o chunks, max over chunks | 0.1242 | 4.08 | length-biased — long pages win the max lottery |
| 2026-06-07 | chunk+meanpool | 200w/40o chunks, mean of chunk vecs (centroid) | 0.2392 | 1.53 | CURRENT shipped; parameter-free, 20 MB index |
| 2026-06-07 20:51 | chunkA:mean | 128w/24o (517k chunks) | 0.2201 |  | mean of chunk vectors (centroid) |
| 2026-06-07 20:51 | chunkA:max | 128w/24o (517k chunks) | 0.1185 |  | reference |
| 2026-06-07 20:51 | chunkA:mean+.5max | 128w/24o (517k chunks) | 0.2061 |  | combine |
| 2026-06-07 20:51 | chunkA:chunk0 | 128w/24o (517k chunks) | 0.2201 |  | lead-only ~baseline |
| 2026-06-07 21:13 | chunkA:mean | 200w/40o (370k chunks) | 0.2392 |  | mean of chunk vectors (centroid) |
| 2026-06-07 21:13 | chunkA:max | 200w/40o (370k chunks) | 0.1242 |  | reference |
| 2026-06-07 21:13 | chunkA:mean+.5max | 200w/40o (370k chunks) | 0.2307 |  | combine |
| 2026-06-07 21:13 | chunkA:chunk0 | 200w/40o (370k chunks) | 0.2244 |  | lead-only ~baseline |
| 2026-06-07 21:29 | chunkA:mean | 300w/50o (249k chunks) | 0.2457 |  | mean of chunk vectors (centroid) |
| 2026-06-07 21:29 | chunkA:max | 300w/50o (249k chunks) | 0.1308 |  | reference |
| 2026-06-07 21:29 | chunkA:mean+.5max | 300w/50o (249k chunks) | 0.2240 |  | combine |
| 2026-06-07 21:29 | chunkA:chunk0 | 300w/50o (249k chunks) | 0.2241 |  | lead-only ~baseline |
| 2026-06-07 21:41 | chunkA:mean | 384w/64o (198k chunks) | 0.2513 |  | mean of chunk vectors (centroid) |
| 2026-06-07 21:41 | chunkA:max | 384w/64o (198k chunks) | 0.1436 |  | reference |
| 2026-06-07 21:41 | chunkA:mean+.5max | 384w/64o (198k chunks) | 0.2389 |  | combine |
| 2026-06-07 21:41 | chunkA:chunk0 | 384w/64o (198k chunks) | 0.2241 |  | lead-only ~baseline |
| 2026-06-07 21:43 | hybrid:dense | 384w/64o + BM25 | 0.2513 |  | w=0 |
| 2026-06-07 21:43 | hybrid:lexical | 384w/64o + BM25 | 0.2238 |  | BM25 only |
| 2026-06-07 21:43 | hybrid:wsum | 384w/64o + BM25 w=0.2 | 0.2735 |  | minmax weighted sum |
| 2026-06-07 21:43 | hybrid:wsum | 384w/64o + BM25 w=0.35 | 0.2605 |  | minmax weighted sum |
| 2026-06-07 21:43 | hybrid:wsum | 384w/64o + BM25 w=0.5 | 0.2623 |  | minmax weighted sum |
| 2026-06-07 21:43 | hybrid:wsum | 384w/64o + BM25 w=0.65 | 0.2316 |  | minmax weighted sum |
| 2026-06-07 21:43 | hybrid:rrf | 384w/64o + BM25 | 0.2502 |  | reciprocal rank fusion |
| 2026-06-07 21:45 | hybrid:dense | 384w/64o + BM25 | 0.2513 |  | w=0 |
| 2026-06-07 21:45 | hybrid:lexical | 384w/64o + BM25 | 0.2238 |  | BM25 only |
| 2026-06-07 21:45 | hybrid:wsum | 384w/64o + BM25 w=0.1 | 0.2601 |  | minmax weighted sum |
| 2026-06-07 21:45 | hybrid:wsum | 384w/64o + BM25 w=0.15 | 0.2694 |  | minmax weighted sum |
| 2026-06-07 21:45 | hybrid:wsum | 384w/64o + BM25 w=0.2 | 0.2735 |  | minmax weighted sum |
| 2026-06-07 21:45 | hybrid:wsum | 384w/64o + BM25 w=0.25 | 0.2728 |  | minmax weighted sum |
| 2026-06-07 21:45 | hybrid:wsum | 384w/64o + BM25 w=0.3 | 0.2671 |  | minmax weighted sum |
| 2026-06-07 21:45 | hybrid:rrf | 384w/64o + BM25 | 0.2502 |  | reciprocal rank fusion |

**Note:** the rows below are on the *new* 29-query public set (commit b4f4b44), not the
original 50, so they are not comparable to the numbers above — only to each other.

| timestamp | experiment | config | NDCG@10 | query_s | notes |
|-----------|------------|--------|---------|---------|-------|
| 2026-06-15 | baseline (new queries) | hybrid 384w/64o + BM25 w=0.2 | 0.3880 | 2.8 | old index, MiniLM truncates each 384w chunk @256 tok (~33% dropped) → coverage gaps |
| 2026-06-15 | fit-window chunks | hybrid 180w/30o + BM25 w=0.2 | **0.4044** | 2.8 | 180w ≈ 235 tok < 256 cap → whole chunk embedded; stride 150 < 180 → no gaps. 392k chunks. **+0.0164** |
