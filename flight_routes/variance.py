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

from math import atan2, cos, radians, sin, sqrt

import numpy as np
import pandas as pd

from .costs import FUEL_KGH, JET_A_EUR_PER_KG, _parse_duration, flight_atc_eur


def haversine_nm(lat1, lon1, lat2, lon2):
    R = 3440.065  # nm
    lat1, lon1, lat2, lon2 = map(radians, [float(lat1), float(lon1), float(lat2), float(lon2)])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def _flight_actual_metrics(firs_grp, pts_grp, eurocontrol_rates):
    """Returns (fir_distances dict nm, actual_duration_h, actual_total_dist_nm) for one flight."""
    f = firs_grp.copy()
    f["entry_dt"] = pd.to_datetime(f["Entry Time"], dayfirst=True)
    f["exit_dt"]  = pd.to_datetime(f["Exit Time"], dayfirst=True)
    airborne = (f[~f["FIR ID"].isin(["TAXI_OUT", "TAXI_IN"])]
                .sort_values("entry_dt").reset_index(drop=True))
    if airborne.empty:
        return {}, np.nan, 0.0

    dur_h = (airborne["exit_dt"].iloc[-1] - airborne["entry_dt"].iloc[0]).total_seconds() / 3600

    p = pts_grp.copy()
    p["time_dt"] = pd.to_datetime(p["Time Over"], dayfirst=True)
    p = (p[(p["time_dt"] >= airborne["entry_dt"].iloc[0]) &
           (p["time_dt"] <= airborne["exit_dt"].iloc[-1])]
         .sort_values("time_dt").reset_index(drop=True))
    if len(p) < 2:
        return {}, dur_h, 0.0

    entries = airborne["entry_dt"].values.astype("datetime64[ns]")
    exits   = airborne["exit_dt"].values.astype("datetime64[ns]")
    fids    = airborne["FIR ID"].values
    lats    = p["Latitude"].values.astype(float)
    lons    = p["Longitude"].values.astype(float)
    times   = p["time_dt"].values.astype("datetime64[ns]")

    def _idx(t):
        for i in range(len(entries)):
            if entries[i] <= t <= exits[i]:
                return i
        return -1

    fir_dists, total_dist = {}, 0.0
    for i in range(len(p) - 1):
        d = haversine_nm(lats[i], lons[i], lats[i + 1], lons[i + 1])
        total_dist += d
        i1, i2 = _idx(times[i]), _idx(times[i + 1])
        if i1 == i2:
            if i1 >= 0:
                fir_dists[fids[i1]] = fir_dists.get(fids[i1], 0) + d
        elif i1 >= 0:
            dt = float((times[i + 1] - times[i]) / np.timedelta64(1, "s"))
            if dt > 0:
                frac = max(0.0, min(1.0,
                    float((exits[i1] - times[i]) / np.timedelta64(1, "s")) / dt))
                fir_dists[fids[i1]] = fir_dists.get(fids[i1], 0) + d * frac
                if i2 >= 0:
                    fir_dists[fids[i2]] = fir_dists.get(fids[i2], 0) + d * (1 - frac)
    # Collapse UIR into FIR: FIR and UIR of the same ANSP are one billing zone
    for fid in list(fir_dists.keys()):
        if fid.endswith("UIR"):
            base_fir = fid[:-3] + "FIR"
            if base_fir in eurocontrol_rates:
                fir_dists[base_fir] = fir_dists.get(base_fir, 0) + fir_dists.pop(fid)
    return fir_dists, dur_h, total_dist


def _total_dist_from_pts(pts_grp):
    """Haversine sum over consecutive filed trajectory points."""
    p = pts_grp.sort_values("Sequence Number").reset_index(drop=True)
    lats = p["Latitude"].values.astype(float)
    lons = p["Longitude"].values.astype(float)
    total = 0.0
    for i in range(len(p) - 1):
        total += haversine_nm(lats[i], lons[i], lats[i + 1], lons[i + 1])
    return round(total, 1)


def build_actual_metrics(df_sample, actual_firs, actual_pts, filed_pts,
                          eurocontrol_rates, ac_col=None):
    """Per-flight realised duration/distance/cost, plus the filed trajectory's total distance.

    df_sample must already have mtow_t (see costs.add_cost_columns).
    """
    ac_col = ac_col or next(c for c in df_sample.columns if "AC Type" in c)
    firs_by_id  = dict(list(actual_firs.groupby("ECTRL ID")))
    pts_by_id   = dict(list(actual_pts.groupby("ECTRL ID")))
    filed_by_id = dict(list(filed_pts.groupby("ECTRL ID")))
    meta = df_sample[["ECTRL ID", ac_col, "mtow_t"]].copy()
    meta["ECTRL ID"] = meta["ECTRL ID"].astype(str)

    records = []
    for _, mrow in meta.iterrows():
        eid = str(mrow["ECTRL ID"])
        if eid not in firs_by_id or eid not in pts_by_id:
            continue
        fir_dists, dur_h, total_dist = _flight_actual_metrics(
            firs_by_id[eid], pts_by_id[eid], eurocontrol_rates
        )
        if not fir_dists or pd.isna(dur_h):
            continue
        ac, mtow = mrow[ac_col], mrow["mtow_t"]
        fkgh = FUEL_KGH.get(ac)
        rep = dict(fir_dists)
        rep["mtow_t"] = mtow
        atc_eur  = flight_atc_eur(rep) if not pd.isna(mtow) else np.nan
        fuel_eur = round(fkgh * dur_h * JET_A_EUR_PER_KG, 2) if fkgh else np.nan
        fuel_kg  = round(fkgh * dur_h, 1) if fkgh else np.nan
        cost_eur = round(atc_eur + fuel_eur, 2) if not (pd.isna(atc_eur) or pd.isna(fuel_eur)) else np.nan
        records.append({
            "ECTRL ID":             int(eid),
            "actual_duration_h":    round(dur_h, 4),
            "actual_total_dist_nm": round(total_dist, 1),
            "actual_atc_eur":       atc_eur,
            "actual_fuel_eur":      fuel_eur,
            "actual_fuel_kg":       fuel_kg,
            "actual_cost_eur":      cost_eur,
            "planned_dist_nm":      _total_dist_from_pts(filed_by_id[eid]) if eid in filed_by_id else np.nan,
        })
    return pd.DataFrame(records)


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
