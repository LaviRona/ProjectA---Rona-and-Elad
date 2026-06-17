"""Shared helper: append one row to RESULTS.md so experiments persist."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

RESULTS_PATH = Path(__file__).resolve().parent.parent / "RESULTS.md"


def log_result(
    experiment: str, config: str, ndcg: float, query_s: float | str = "", notes: str = ""
) -> None:
    """Append a result row. Creates the table header if the file is missing."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    qs = f"{query_s:.2f}" if isinstance(query_s, (int, float)) else str(query_s)
    row = f"| {ts} | {experiment} | {config} | {ndcg:.4f} | {qs} | {notes} |\n"
    if not RESULTS_PATH.exists():
        RESULTS_PATH.write_text(
            "# Section B — Experiment results log (mean NDCG@10 on 50 public queries)\n\n"
            "| timestamp | experiment | config | NDCG@10 | query_s | notes |\n"
            "|-----------|------------|--------|---------|---------|-------|\n"
        )
    with RESULTS_PATH.open("a", encoding="utf-8") as f:
        f.write(row)
    print(f"[logged] {experiment} | {config} | NDCG@10={ndcg:.4f}")
