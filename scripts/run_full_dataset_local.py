"""Local reproduction of notebook cells 29/30/31/32/32b/32c (full-dataset extension).

Runs the part of the Colab pipeline that has never completed end to end,
against the local data/raw/ files (default FLIGHT_ROUTES_RAW_DIR/CACHE_DIR,
no Drive mount needed). Kept out of clustering.ipynb on purpose: that
notebook is the Colab source of truth (cell 0 clones from GitHub, cell 1
mounts Drive) and isn't meant to run unmodified locally.
"""
import gc
import time
from pathlib import Path

import pandas as pd

from flight_routes import data as frdata
from flight_routes.features import fir_columns
from flight_routes.clustering import cluster_full_dataset
from flight_routes.query import build_full_summary
from flight_routes.costs import add_cost_columns, rate_firs_in, EUROCONTROL_RATES
from flight_routes.variance import build_actual_metrics_full_dataset, compute_centroid_deltas, compute_self_deltas
from flight_routes.carriers import carrier_coverage, summarise_self_deltas_by_carrier

t0 = time.time()
def log(msg):
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)

log(f"raw_dir={frdata.raw_dir()}")
log(f"cache_dir={frdata.cache_dir()}")

RESULTS_DIR = Path(__file__).resolve().parent.parent / "docs" / "full_dataset_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Checkpoint so a kill/crash after this point doesn't repeat the ~70min clustering step.
CHECKPOINT_DIR = frdata.cache_dir() / "_local_run_checkpoint"
CHECKPOINT_DF = CHECKPOINT_DIR / "df_full_clustered.parquet"
CHECKPOINT_DIAG = CHECKPOINT_DIR / "clust_diagnostic.parquet"

if CHECKPOINT_DF.exists() and CHECKPOINT_DIAG.exists():
    df_full = pd.read_parquet(CHECKPOINT_DF)
    clust_diagnostic = pd.read_parquet(CHECKPOINT_DIAG)
    log(f"resumed from checkpoint: {len(df_full):,} flights, already clustered")
else:
    # Cell 29 - load / build full qualifying dataset (validated threshold: >=50 flights/pair)
    MIN_FLIGHTS_PER_OD = 50
    df_full = frdata.load_full_dataset(min_flights_per_od=MIN_FLIGHTS_PER_OD)
    _fir_all_cols = fir_columns(df_full)
    n_pairs_full = df_full.groupby(["ADEP", "ADES"]).ngroups
    log(f"df_full loaded: {len(df_full):,} flights, {n_pairs_full:,} qualifying pairs, {len(_fir_all_cols)} FIR cols")

    # Cell 30 - three-layer clustering on all qualifying pairs
    full_labels, clust_diagnostic = cluster_full_dataset(df_full, _fir_all_cols)
    df_full["cluster_kmeans"] = full_labels
    log("cluster_full_dataset done")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    df_full.to_parquet(CHECKPOINT_DF, index=False)
    clust_diagnostic.to_parquet(CHECKPOINT_DIAG, index=False)
    log(f"checkpoint written to {CHECKPOINT_DIR}")

print(clust_diagnostic["n_clusters"].value_counts().sort_index().rename("n_pairs").to_string())

# Cell 31 - band diagnostic (no plot, just the table)
bands = [(30, 50), (50, 75), (75, 100), (100, 150), (150, 200), (200, 300), (300, 500), (500, 9999)]
print(f'\n{"Band":>12}  {"Pairs":>7}  {"k=1":>7}  {"k>=2":>7}  {"% k>=2":>7}')
print("-" * 48)
for lo, hi in bands:
    sub = clust_diagnostic[(clust_diagnostic["n_flights"] >= lo) & (clust_diagnostic["n_flights"] < hi)]
    if sub.empty:
        continue
    k2p = (sub["n_clusters"] >= 2).sum()
    lbl = f"{lo}-{hi}" if hi < 9999 else f"{lo}+"
    print(f'{lbl:>12}  {len(sub):>7,}  {(sub["n_clusters"]==1).sum():>7,}  {k2p:>7,}  {k2p/len(sub)*100:>6.1f}%')

# Cell 32 - full_summary: vectorised cost computation across all pairs
full_summary, full_summary_ac = build_full_summary(df_full)
n_od = full_summary.groupby(["ADEP", "ADES"]).ngroups
n_alt = (full_summary.groupby(["ADEP", "ADES"])["cluster_kmeans"].nunique() >= 2).sum()
MIN_N_AC = 10
n_ac_specific = (full_summary_ac["n_flights_ac"] >= MIN_N_AC).sum()
log(f"full_summary: {len(full_summary):,} cluster rows, {n_od:,} O-D pairs")
print(f"Pairs with alternatives (k>=2): {n_alt:,} ({n_alt / n_od * 100:.1f}%)")
print(f"full_summary_ac: {len(full_summary_ac):,} rows, {n_ac_specific:,} meet MIN_N_AC={MIN_N_AC}")

del full_summary, full_summary_ac, clust_diagnostic
gc.collect()

# Cell 32b - full-dataset variance: actual vs planned, Type 1 and Type 2
df_full = add_cost_columns(df_full)
log("add_cost_columns done")
gc.collect()

actual_metrics_full = build_actual_metrics_full_dataset(
    df_full, EUROCONTROL_RATES, batch_size=10_000, force_rebuild=False
)
log(f"actual_metrics_full: {len(actual_metrics_full):,} flights")

result = compute_centroid_deltas(df_full, actual_metrics_full, rate_firs_in(df_full))
df_compared_full = result["df_compared"]
error_summary_full = result["error_summary"]
error_summary_pooled_full = result["error_summary_pooled"]
log("compute_centroid_deltas done")
error_summary_full.to_csv(RESULTS_DIR / "error_summary_full.csv", index=False)
error_summary_pooled_full.to_csv(RESULTS_DIR / "error_summary_pooled_full.csv", index=False)

df_compared_full, error_summary_self_full, error_summary_self_pair_full = compute_self_deltas(df_compared_full)
log("compute_self_deltas done")
print(error_summary_self_pair_full.head(10).to_string())
error_summary_self_full.to_csv(RESULTS_DIR / "error_summary_self_full.csv", index=False)
error_summary_self_pair_full.to_csv(RESULTS_DIR / "error_summary_self_pair_full.csv", index=False)

# Cell 32c - self-delta variance by carrier type
coverage = carrier_coverage(df_compared_full)
print(coverage)
carrier_summary = summarise_self_deltas_by_carrier(df_compared_full)
print(carrier_summary)
coverage.to_csv(RESULTS_DIR / "carrier_coverage.csv", index=False)
carrier_summary.to_csv(RESULTS_DIR / "carrier_summary_full.csv", index=False)

log(f"DONE, CSVs written to {RESULTS_DIR}")
