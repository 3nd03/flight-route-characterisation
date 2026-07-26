"""Investigate why LTFM-origin O-D pairs show unusually wide planned-vs-realised
distance spread (Report Future Work item). Not part of the shipped pipeline -
one-off exploratory analysis using data/functions already validated elsewhere.

Step 1: look at per-flight deltas for the 3 worst pairs, not just the pair-level
std, to see whether the spread is a real bimodal/clustered pattern or just noise.
"""
import time

import pandas as pd

from flight_routes import data as frdata
from flight_routes.costs import add_cost_columns, rate_firs_in, EUROCONTROL_RATES
from flight_routes.variance import build_actual_metrics_full_dataset, compute_centroid_deltas, compute_self_deltas

t0 = time.time()
def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)

CHECKPOINT_DF = frdata.cache_dir() / "_local_run_checkpoint" / "df_full_clustered.parquet"
df_full = pd.read_parquet(CHECKPOINT_DF)
log(f"df_full loaded from checkpoint: {len(df_full):,} flights")

df_full = add_cost_columns(df_full)
log("add_cost_columns done")

actual_metrics_full = build_actual_metrics_full_dataset(
    df_full, EUROCONTROL_RATES, batch_size=10_000, force_rebuild=False
)
log(f"actual_metrics_full loaded (cached): {len(actual_metrics_full):,} flights")

result = compute_centroid_deltas(df_full, actual_metrics_full, rate_firs_in(df_full))
df_compared_full = result["df_compared"]
log("compute_centroid_deltas done")

df_compared_full, _, _ = compute_self_deltas(df_compared_full)
log("compute_self_deltas done")

PAIRS = [("LTFM", "OOMS"), ("LTFM", "EBBR"), ("LTFM", "LMML")]
cols = ["ECTRL ID", "ADEP", "ADES", "AC Operator", "cluster_kmeans",
        "actual_total_dist_nm", "planned_dist_nm",
        "delta_dist_nm_self", "delta_duration_h_self", "delta_fuel_kg_self"]

subset = df_compared_full[
    df_compared_full[["ADEP", "ADES"]].apply(tuple, axis=1).isin(PAIRS)
][cols].dropna(subset=["delta_dist_nm_self"]).sort_values(["ADEP", "ADES", "delta_dist_nm_self"])

out_path = frdata.cache_dir() / "_ltfm_investigation_flights.csv"
subset.to_csv(out_path, index=False)
log(f"wrote {len(subset)} flights to {out_path}")

for pair in PAIRS:
    sub = subset[(subset["ADEP"] == pair[0]) & (subset["ADES"] == pair[1])]
    print(f"\n=== {pair[0]}-{pair[1]} (n={len(sub)}) ===")
    print(sub["delta_dist_nm_self"].describe().to_string())
    print("Value counts by 50nm bucket:")
    print(pd.cut(sub["delta_dist_nm_self"], bins=range(-400, 401, 50)).value_counts().sort_index().to_string())

log("DONE")
