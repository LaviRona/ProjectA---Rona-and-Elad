"""Throwaway runner: evaluate vector_index_new.VectorIndex on a public scenario."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from evaluation import get_local_baseline_stats, run_scenario, SCENARIO_WALL_LIMIT_SEC
from naive_vector_index import VectorIndex as Naive
from vector_index_new import VectorIndex as Mine

p = argparse.ArgumentParser()
p.add_argument("--scenario", type=int, required=True)
p.add_argument("--recalibrate-baseline", action="store_true")
a = p.parse_args()

sd = ROOT / "data" / "public" / f"scenario_{a.scenario:02d}"
print("Measuring local naive baseline...")
bs = get_local_baseline_stats(sd, Naive, recalibrate=a.recalibrate_baseline)
print(f"baseline_dynamic={bs['baseline_dynamic']:.4f}s initial={bs['baseline_initial']:.4f}s")

s = run_scenario(sd, Mine, baseline_time=bs["baseline_dynamic"])
print(f"--- scenario {a.scenario:02d} (vector_index_new) ---")
for key in ("insert_score", "delete_score", "search_score", "functional_score",
            "runtime_multiplier", "final_score", "initial_time", "dynamic_time",
            "scenario_wall_time", "speed_ratio"):
    print(f"{key}={s[key]:.4f}")
print(f"wall_timeout={bool(s['wall_timeout'])} (limit={SCENARIO_WALL_LIMIT_SEC:.0f}s)")
