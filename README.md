# Section B — Retrieval pipeline

End-to-end retrieval over the Wikipedia-style corpus. `run(queries)` returns, for
each query, a ranked list of `page_id` (most relevant first); only the top 10 are
scored (mean NDCG@10).

## Pipeline

| Stage | File | Method |
|-------|------|--------|
| Chunk | `chunk.py` | Overlapping fixed-size **word windows** (200 words, 40 overlap), title prepended to each chunk, capped at 60 chunks/page. |
| Embed | `embed.py` | `sentence-transformers/all-MiniLM-L6-v2`, L2-normalized, 384-dim. |
| Index | `index.py` | Each page = **mean of its chunk vectors** (one un-normalized centroid per page), stored float16. |
| Retrieve | `retrieve.py` | Exact dot-product search over page centroids, top-10. |

**Why chunk, then average?** Articles are long (median ~1.2k words; ~75% exceed
256 words) while MiniLM truncates at 256 word-pieces — embedding a whole page
only captures its opening. We chunk so the *whole* article is embedded, then
represent each page by the **mean of its chunk vectors**. Dotting a query with
that mean equals the mean similarity over the page's passages (mean-pooling).

We measured three aggregations on the 50 public queries (`scripts/diagnose.py`):

| Aggregation | NDCG@10 |
|-------------|---------|
| Whole-page (truncated) baseline | 0.224 |
| Max over chunks | 0.124 ❌ (length-biased: long pages get many chances to match) |
| **Mean over chunks (chosen)** | **0.239** ✅ |

Mean-pooling removes the length bias and is parameter-free, so it generalizes to
the hidden set (unlike a tuned length penalty, which swung sharply with its
coefficient on only 50 queries).

## Setup

```bash
cd SectionB
pip install -r requirements.txt
```

The corpus must be reachable at `data/Wikipedia Entries/` and queries at
`data/public_queries.json` (see `utils.py`).

## Build index (offline, not timed — your machine only)

Run once locally to create `artifacts/`. **These files are submitted**; staff do
not rebuild the index at grading time.

```bash
python scripts/build_index.py
```

### Artifacts (`artifacts/`)

| File | Format |
|------|--------|
| `index_vectors.npy` | `(num_pages, 384)` float16 matrix; row `i` is the mean chunk vector for `page_ids[i]`. ~20 MB. |
| `index_meta.json` | `{ page_ids: [...], model: "...", num_vectors: N, level: "page_centroid (mean of chunk vectors)" }` — `page_ids[i]` is the page for row `i` (one row per page). |

`index_vectors.npy` is ~20 MB, so it commits to Git directly (no LFS needed).

## Public self-test

After building, verify a fresh run loads the submitted artifacts (no rebuild):

```bash
python scripts/eval_public.py
```

Prints mean NDCG@10 on the 50 public queries.

## Experiments

`python scripts/diagnose.py` reproduces the aggregation comparison above on the
50 public queries, reusing the built artifacts (no rebuild). Used to choose
mean-pooling and to generate the video's empirical results.

## Tuning knobs

`CHUNK_WORDS`, `OVERLAP_WORDS`, `MAX_CHUNKS_PER_PAGE` in `chunk.py`.

## Submit

Public GitHub repo with this code, the **required** `artifacts/`, and a link to the
presentation video. See the assignment PDF for video and grading details.

## Video Presentation

[Watch the Section B presentation video](Video/presentation.mp4)
