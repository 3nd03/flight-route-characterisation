"""Planned-vs-realised variance: two distinct comparisons, both aggregated to
mean/std/5th-95th percentile so either can be sampled from directly.

1. build_actual_metrics + compute_centroid_deltas - each flight's realised
   cost/duration/distance against the mean of its own cluster's planned
   values (the cluster representative). This is the noise a simulator needs
   to add around that single stored representative.
2. compute_self_deltas - each flight's realised trajectory against its own
   filed plan, independent of clustering. Measures how reliably route
   planning itself predicts what is actually flown; comparable across
   clusters and O-D pairs.
"""

import gc
import shutil

import numpy as np
import pandas as pd

from . import cache_db, data
from .costs import FUEL_KGH, JET_A_EUR_PER_KG, MTOW_TONNES, _parse_duration, flight_atc_eur


def haversine_nm(lat1, lon1, lat2, lon2):
    """Great-circle distance in nautical miles. Works elementwise on scalars or numpy arrays."""
    R = 3440.065  # nm
    lat1, lon1, lat2, lon2 = np.radians(lat1), np.radians(lon1), np.radians(lat2), np.radians(lon2)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def _consecutive_segment_distances(df, id_col, lat_col, lon_col):
    """Haversine distance from each row to the next row of the same flight.

    df must already be sorted by (id_col, <time/sequence col>). Returns a
    Series aligned to df's index; the last row of each flight (and rows
    followed by a different flight) get NaN, since there is no "next" point
    within the same flight to measure to.
    """
    same_next = df[id_col].shift(-1) == df[id_col]
    d = haversine_nm(
        df[lat_col].to_numpy(dtype=float), df[lon_col].to_numpy(dtype=float),
        df[lat_col].shift(-1).to_numpy(dtype=float), df[lon_col].shift(-1).to_numpy(dtype=float),
    )
    return pd.Series(d, index=df.index).where(same_next)


def _remap_uir_to_fir(fir_id, eurocontrol_rates):
    """FIR and UIR of the same ANSP are one billing zone - collapse UIR into FIR
    only when that FIR actually has a rate entry, otherwise leave it as-is."""
    if fir_id.endswith("UIR"):
        base = fir_id[:-3] + "FIR"
        if base in eurocontrol_rates:
            return base
    return fir_id


def build_actual_metrics(df_sample, actual_firs, actual_pts, filed_pts,
                          eurocontrol_rates, ac_col=None):
    """Per-flight realised duration/distance/cost, plus the filed trajectory's total distance.

    Vectorised across all flights at once (no per-flight Python loop): each
    trajectory point is matched to its FIR window with a single pd.merge_asof
    (by=ECTRL ID), and consecutive-point distances/FIR attribution are done
    with groupby/shift instead of a nested per-flight, per-point loop. See
    tests/test_variance.py for edge-case validation against the original
    per-flight-loop implementation (same-FIR segments, FIR-crossing segments
    with time-proportional splitting, gaps with no FIR match, UIR/FIR
    consolidation).

    df_sample must already have mtow_t (see costs.add_cost_columns).
    """
    ac_col = ac_col or next(c for c in df_sample.columns if "AC Type" in c)

    # --- per-flight airborne window (first entry, last exit of the sorted-by-entry sequence) ---
    firs = actual_firs[~actual_firs["FIR ID"].isin(["TAXI_OUT", "TAXI_IN"])].copy()
    # format="mixed": some rows lack zero-padding (e.g. single-digit day/month),
    # so pandas can't infer one fixed format across the whole column and falls
    # back to the much slower per-element dateutil parser with just dayfirst=True.
    # "mixed" uses pandas' own faster per-element parser instead, without
    # requiring - or risking erroring on - a single strict width.
    firs["entry_dt"] = pd.to_datetime(firs["Entry Time"], format="mixed", dayfirst=True)
    firs["exit_dt"]  = pd.to_datetime(firs["Exit Time"], format="mixed", dayfirst=True)
    firs = firs.sort_values(["ECTRL ID", "entry_dt"])

    bounds = firs.groupby("ECTRL ID").agg(
        window_start=("entry_dt", "first"), window_end=("exit_dt", "last")
    )
    duration_h = (bounds["window_end"] - bounds["window_start"]).dt.total_seconds() / 3600

    # --- actual trajectory points, restricted to each flight's airborne window ---
    pts = actual_pts.copy()
    pts["time_dt"] = pd.to_datetime(pts["Time Over"], format="mixed", dayfirst=True)
    pts = pts.merge(bounds, on="ECTRL ID", how="inner")
    pts = pts[(pts["time_dt"] >= pts["window_start"]) & (pts["time_dt"] <= pts["window_end"])]

    # merge_asof with by= still requires the "on" column sorted *globally* (not just within
    # each group), so sort purely by time/entry_dt here; re-sort by (ECTRL ID, time_dt) after,
    # since the consecutive-segment logic below needs grouped-then-time order instead.
    matched = pd.merge_asof(
        pts[["ECTRL ID", "time_dt", "Latitude", "Longitude"]].sort_values("time_dt"),
        firs[["ECTRL ID", "entry_dt", "exit_dt", "FIR ID"]].sort_values("entry_dt"),
        left_on="time_dt", right_on="entry_dt", by="ECTRL ID", direction="backward",
    )
    matched["FIR ID"] = matched["FIR ID"].where(matched["time_dt"] <= matched["exit_dt"])
    matched = matched.sort_values(["ECTRL ID", "time_dt"]).reset_index(drop=True)

    # --- consecutive-point distance and start/end FIR for each segment ---
    same_next = matched["ECTRL ID"].shift(-1) == matched["ECTRL ID"]
    matched["seg_dist"] = _consecutive_segment_distances(matched, "ECTRL ID", "Latitude", "Longitude")
    matched["fir_end"]  = matched["FIR ID"].shift(-1).where(same_next)
    matched["t_end"]    = matched["time_dt"].shift(-1).where(same_next)

    seg = matched.dropna(subset=["seg_dist"]).copy()
    total_dist = seg.groupby("ECTRL ID")["seg_dist"].sum()  # every segment counts, regardless of FIR

    fir_start_valid = seg["FIR ID"].notna()
    same_fir  = fir_start_valid & (seg["FIR ID"] == seg["fir_end"])
    cross_fir = fir_start_valid & ~same_fir  # segment leaves its starting FIR (into another, or a gap)

    dt_total   = (seg["t_end"] - seg["time_dt"]).dt.total_seconds()
    dt_to_exit = (seg["exit_dt"] - seg["time_dt"]).dt.total_seconds()
    frac_start = (dt_to_exit / dt_total).clip(lower=0.0, upper=1.0)

    parts = [seg.loc[same_fir, ["ECTRL ID", "FIR ID"]].assign(dist=seg.loc[same_fir, "seg_dist"])]
    parts.append(seg.loc[cross_fir, ["ECTRL ID", "FIR ID"]].assign(
        dist=seg.loc[cross_fir, "seg_dist"] * frac_start[cross_fir]))
    # the remainder only goes to the ending FIR if there is one (not a FIR-to-gap transition,
    # matching the original: distance is dropped, not assigned, when the segment ends in a gap)
    cross_to_fir = cross_fir & seg["fir_end"].notna()
    parts.append(seg.loc[cross_to_fir, ["ECTRL ID", "fir_end"]].rename(columns={"fir_end": "FIR ID"}).assign(
        dist=seg.loc[cross_to_fir, "seg_dist"] * (1 - frac_start[cross_to_fir])))

    fir_dist_long = pd.concat(parts, ignore_index=True)
    fir_dist_long["FIR ID"] = fir_dist_long["FIR ID"].apply(lambda f: _remap_uir_to_fir(f, eurocontrol_rates))
    fir_dist_long = fir_dist_long.groupby(["ECTRL ID", "FIR ID"])["dist"].sum().reset_index()
    fir_wide = fir_dist_long.pivot(index="ECTRL ID", columns="FIR ID", values="dist").fillna(0)

    # --- filed trajectory's total distance (planned_dist_nm) ---
    fp = filed_pts.sort_values(["ECTRL ID", "Sequence Number"]).reset_index(drop=True)
    fp["seg_dist"] = _consecutive_segment_distances(fp, "ECTRL ID", "Latitude", "Longitude")
    planned_dist = fp.groupby("ECTRL ID")["seg_dist"].sum()

    # --- assemble per-flight metrics; skip flights missing FIR/point data entirely (inner joins) ---
    meta = df_sample[["ECTRL ID", ac_col, "mtow_t"]].copy()
    meta["ECTRL ID"] = meta["ECTRL ID"].astype(str)

    out = meta.merge(fir_wide, left_on="ECTRL ID", right_index=True, how="inner")
    out = out.merge(duration_h.rename("actual_duration_h"), left_on="ECTRL ID", right_index=True, how="inner")
    out = out.merge(total_dist.rename("actual_total_dist_nm"), left_on="ECTRL ID", right_index=True, how="inner")
    out = out.merge(planned_dist.rename("planned_dist_nm"), left_on="ECTRL ID", right_index=True, how="left")
    out = out[out["actual_duration_h"].notna()]

    fir_cols = list(fir_wide.columns)

    def _row_atc(row):
        rep = {fir: row[fir] for fir in fir_cols}
        rep["mtow_t"] = row["mtow_t"]
        return flight_atc_eur(rep) if pd.notna(row["mtow_t"]) else np.nan

    out["actual_atc_eur"] = out.apply(_row_atc, axis=1)
    fuel_kgh = out[ac_col].map(FUEL_KGH)
    out["actual_fuel_kg"]  = (fuel_kgh * out["actual_duration_h"]).round(1)
    out["actual_fuel_eur"] = (out["actual_fuel_kg"] * JET_A_EUR_PER_KG).round(2)
    out["actual_cost_eur"] = (out["actual_atc_eur"] + out["actual_fuel_eur"]).round(2)
    out["actual_duration_h"]    = out["actual_duration_h"].round(4)
    out["actual_total_dist_nm"] = out["actual_total_dist_nm"].round(1)
    out["planned_dist_nm"]      = out["planned_dist_nm"].round(1)
    out["ECTRL ID"] = out["ECTRL ID"].astype(int)

    return out[["ECTRL ID", "actual_duration_h", "actual_total_dist_nm",
                "actual_atc_eur", "actual_fuel_eur", "actual_fuel_kg",
                "actual_cost_eur", "planned_dist_nm"]].reset_index(drop=True)


def build_actual_metrics_full_dataset(df_full, eurocontrol_rates, ac_col=None,
                                       batch_size=20_000, force_rebuild=False,
                                       progress_every=5, month=data.DEFAULT_MONTH):
    """build_actual_metrics for the whole dataset (536,520+ flights), memory-bounded.

    data.load_actual_and_filed materialises every matching row of all three
    raw files in RAM at once - fine at training-sample scale (~1,650 flights,
    a tiny fraction of each file) but not at full-dataset scale, where the
    row filter survives roughly 60% of each multi-GB file.

    Instead: split flights into batch_size-sized batches, partition each raw
    file to per-batch CSVs on disk with a single chunked pass per file
    (data._partition_by_batch - never more than one chunk in RAM), then run
    build_actual_metrics per batch and keep only its small per-flight output,
    discarding the batch's raw rows before moving to the next. Mirrors
    clustering.cluster_full_dataset's streaming + gc.collect() pattern: disk
    and time are cheap on Colab's free tier, RAM is the scarce resource.

    Cached as a single "actual_metrics_full" table (see cache_db.py) once
    computed, since recomputing over all batches is the expensive part.
    """
    cache = data.cache_dir()
    if not force_rebuild and cache_db.has_table(cache, "actual_metrics_full"):
        return cache_db.read_table(cache, "actual_metrics_full")

    ac_col = ac_col or next(c for c in df_full.columns if "AC Type" in c)
    # No defensive df_full.copy() here: at full-dataset scale df_full's 300+ float
    # columns get added incrementally (FIR cols from the raw join, mtow_t/atc_eur/
    # fuel_eur/cost_eur from add_cost_columns), leaving them as separate unconsolidated
    # blocks - .copy() forces pandas to consolidate them into one new contiguous
    # (n_float_cols, n_rows) array, a 1.24 GiB allocation on the real dataset, on top
    # of everything else already resident. The two mutations below are column-level
    # and idempotent (str cast, additive column), so they're applied in place instead.
    df_full["ECTRL ID"] = df_full["ECTRL ID"].astype(str)
    if "mtow_t" not in df_full.columns:
        df_full["mtow_t"] = df_full[ac_col].map(MTOW_TONNES)

    ids = df_full["ECTRL ID"].unique()
    id_to_batch = {eid: i // batch_size for i, eid in enumerate(ids)}
    n_batches = max(id_to_batch.values()) + 1

    raw = data.raw_dir()
    tmp_dir = cache / "_variance_batches"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    try:
        firs_paths = data._partition_by_batch(data._firs_actual_path(raw, month), id_to_batch, tmp_dir, "firs")
        pts_paths = data._partition_by_batch(data._points_actual_path(raw, month), id_to_batch, tmp_dir, "pts")
        filed_paths = data._partition_by_batch(data._points_filed_path(raw, month), id_to_batch, tmp_dir, "filed")

        batch_num = df_full["ECTRL ID"].map(id_to_batch)
        empty_filed = pd.DataFrame(columns=["ECTRL ID", "Sequence Number", "Latitude", "Longitude"])
        results = []
        for b in range(n_batches):
            if b not in firs_paths or b not in pts_paths:
                continue
            batch_sample = df_full[batch_num == b]
            actual_firs = pd.read_csv(firs_paths[b], dtype={"ECTRL ID": str})
            actual_pts  = pd.read_csv(pts_paths[b], dtype={"ECTRL ID": str})
            filed_pts   = pd.read_csv(filed_paths[b], dtype={"ECTRL ID": str}) if b in filed_paths else empty_filed

            results.append(build_actual_metrics(
                batch_sample, actual_firs, actual_pts, filed_pts, eurocontrol_rates, ac_col=ac_col
            ))

            del actual_firs, actual_pts, filed_pts, batch_sample
            if (b + 1) % progress_every == 0 or b == n_batches - 1:
                gc.collect()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    out = pd.concat(results, ignore_index=True)
    cache_db.write_table(cache, "actual_metrics_full", out, index_cols=["ECTRL ID"])
    return out


def _summarise(df, group_cols, delta_cols):
    rows = []
    for key, grp in df.dropna(subset=delta_cols).groupby(group_cols):
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key_tuple))
        row["n"] = len(grp)
        for col in delta_cols:
            row[f"{col}_mean"] = round(grp[col].mean(), 2)
            row[f"{col}_std"]  = round(grp[col].std(), 2)
            row[f"{col}_p5"]   = round(grp[col].quantile(0.05), 2)
            row[f"{col}_p95"]  = round(grp[col].quantile(0.95), 2)
        rows.append(row)
    return pd.DataFrame(rows)


DELTA_COLS        = ["delta_dist_nm", "delta_duration_h", "delta_fuel_kg"]
DELTA_COLS_SELF   = ["delta_dist_nm_self", "delta_duration_h_self", "delta_fuel_kg_self"]


def compute_centroid_deltas(df_sample, df_actual, rate_firs, ac_col=None):
    """Type 1: each flight vs its cluster+AC-type centroid.

    df_sample must already have cost_eur/fuel_eur (see costs.add_cost_columns)
    and a cluster_kmeans column (see clustering.cluster_od_3layer + merge_similar_clusters).

    Returns dict with df_compared, l3_centroids, l3_centroids_pooled,
    error_summary, error_summary_pooled.
    """
    ac_col = ac_col or next(c for c in df_sample.columns if "AC Type" in c)
    df_sample = df_sample.copy()
    df_sample["ECTRL ID"] = df_sample["ECTRL ID"].astype(int)
    df_sample["planned_duration_h"] = df_sample["Duration_Hours"].apply(_parse_duration)
    df_sample["planned_fuel_kg"]    = (df_sample["fuel_eur"] / JET_A_EUR_PER_KG).round(1)

    df_compared = df_sample.merge(df_actual, on="ECTRL ID", how="left")
    centroid_feats = ["planned_dist_nm", "planned_duration_h", "planned_fuel_kg", "cost_eur"]

    l3_centroids = (
        df_compared.dropna(subset=centroid_feats)
        .groupby(["ADEP", "ADES", "cluster_kmeans", ac_col])[centroid_feats]
        .agg(
            centroid_dist_nm=("planned_dist_nm", "mean"),
            centroid_duration_h=("planned_duration_h", "mean"),
            centroid_fuel_kg=("planned_fuel_kg", "mean"),
            centroid_cost_eur=("cost_eur", "mean"),
            n_l3=("planned_dist_nm", "count"),
        )
        .round({"centroid_dist_nm": 1, "centroid_duration_h": 3,
                "centroid_fuel_kg": 1, "centroid_cost_eur": 2})
        .reset_index()
    )

    df_compared = df_compared.merge(
        l3_centroids[["ADEP", "ADES", "cluster_kmeans", ac_col,
                      "centroid_dist_nm", "centroid_duration_h",
                      "centroid_fuel_kg", "centroid_cost_eur"]],
        on=["ADEP", "ADES", "cluster_kmeans", ac_col], how="left",
    )
    # delta_cost_eur intentionally excluded: planned FIR double-counting inflates
    # centroid_cost_eur relative to actual_cost_eur - see docs/report_draft.md Discussion.
    df_compared["delta_dist_nm"]    = (df_compared["actual_total_dist_nm"] - df_compared["centroid_dist_nm"]).round(1)
    df_compared["delta_duration_h"] = (df_compared["actual_duration_h"] - df_compared["centroid_duration_h"]).round(4)
    df_compared["delta_fuel_kg"]    = (df_compared["actual_fuel_kg"] - df_compared["centroid_fuel_kg"]).round(1)

    error_summary = _summarise(df_compared, ["ADEP", "ADES", "cluster_kmeans", ac_col], DELTA_COLS)

    pooled_agg = {
        "centroid_dist_nm":    ("planned_dist_nm", "mean"),
        "centroid_duration_h": ("planned_duration_h", "mean"),
        "n_l3":                ("planned_dist_nm", "count"),
    }
    pooled_agg.update({f"mean_{f}": (f, "mean") for f in rate_firs if f in df_compared.columns})

    l3_centroids_pooled = (
        df_compared.dropna(subset=centroid_feats)
        .groupby(["ADEP", "ADES", "cluster_kmeans"])
        .agg(**pooled_agg)
        .round({"centroid_dist_nm": 1, "centroid_duration_h": 3})
        .reset_index()
    )

    df_compared = df_compared.merge(
        l3_centroids_pooled[["ADEP", "ADES", "cluster_kmeans", "centroid_dist_nm", "centroid_duration_h"]],
        on=["ADEP", "ADES", "cluster_kmeans"], how="left", suffixes=("", "_pooled"),
    )
    df_compared["delta_dist_nm_pooled"]    = (df_compared["actual_total_dist_nm"] - df_compared["centroid_dist_nm_pooled"]).round(1)
    df_compared["delta_duration_h_pooled"] = (df_compared["actual_duration_h"] - df_compared["centroid_duration_h_pooled"]).round(4)

    error_summary_pooled = _summarise(
        df_compared, ["ADEP", "ADES", "cluster_kmeans"],
        ["delta_dist_nm_pooled", "delta_duration_h_pooled"],
    )

    return {
        "df_compared": df_compared,
        "l3_centroids": l3_centroids,
        "l3_centroids_pooled": l3_centroids_pooled,
        "error_summary": error_summary,
        "error_summary_pooled": error_summary_pooled,
    }


def compute_self_deltas(df_compared):
    """Type 2: each flight vs its own filed plan, independent of cluster.

    Returns (df_compared_with_self_deltas, error_summary_self, error_summary_self_by_pair).
    """
    df_compared = df_compared.copy()
    df_compared["delta_dist_nm_self"]    = (df_compared["actual_total_dist_nm"] - df_compared["planned_dist_nm"]).round(1)
    df_compared["delta_duration_h_self"] = (df_compared["actual_duration_h"] - df_compared["planned_duration_h"]).round(4)
    df_compared["delta_fuel_kg_self"]    = (df_compared["actual_fuel_kg"] - df_compared["planned_fuel_kg"]).round(1)

    error_summary_self      = _summarise(df_compared, ["ADEP", "ADES", "cluster_kmeans"], DELTA_COLS_SELF)
    error_summary_self_pair = _summarise(df_compared, ["ADEP", "ADES"], DELTA_COLS_SELF)
    return df_compared, error_summary_self, error_summary_self_pair
