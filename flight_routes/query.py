"""The public, simulator-facing API: given an aircraft type and O-D pair,
return ranked route alternatives with predicted cost, duration, and (where
available) error bounds.

Two query functions, because they draw on different scopes of the pipeline:

- predict_route_options() - full dataset (~4,500+ O-D pairs), cost/duration
  only. This is what full_summary.csv (the small, git-tracked "query-ready"
  output) is built for - no raw data needed to call this.
- query_route_profile() - the three training pairs only, but with the Type 1
  error envelope (delta_*_mean/std/p5/p95) attached, since that comparison
  has only been run on the training/validation sample so far.
"""

import numpy as np
import pandas as pd

from .costs import (
    CAD_EUR,
    EUROCONTROL_RATES,
    FUEL_KGH,
    JET_A_EUR_PER_KG,
    MTOW_TONNES,
    NAV_CANADA_OCEANIC,
    NAV_CANADA_R,
    _parse_duration,
    flight_atc_eur,
    rate_firs_in,
)

MIN_N_AC = 10


def build_full_summary(df_full, ac_col=None, min_n_ac=MIN_N_AC):
    """Vectorised cost + per-cluster and per-(cluster, aircraft type) aggregation.

    df_full must already have a cluster_kmeans column (see
    clustering.cluster_full_dataset). Returns (full_summary, full_summary_ac) -
    full_summary is what predict_route_options() queries against; it's small
    enough (one row per O-D-pair/cluster) to commit to git and ship with the
    library, unlike df_full itself.
    """
    df_full = df_full.copy()
    ac_col = ac_col or next(c for c in df_full.columns if "AC Type" in c)
    rate_firs = rate_firs_in(df_full)

    df_full["mtow_t"]     = df_full[ac_col].map(MTOW_TONNES)
    df_full["duration_h"] = df_full["Duration_Hours"].apply(_parse_duration)

    def _atc_vec(df):
        mtow = df["mtow_t"]
        total = pd.Series(0.0, index=df.index)
        base_totals, base_rate_map = {}, {}
        for fir, rate in EUROCONTROL_RATES.items():
            if fir not in df.columns:
                continue
            base = fir[:-3]
            base_totals[base] = base_totals.get(base, pd.Series(0.0, index=df.index)) + df[fir].fillna(0)
            base_rate_map.setdefault(base, rate)
        for base, dist_nm_col in base_totals.items():
            dist_km = dist_nm_col * 1.852
            total += (dist_km / 100) * np.sqrt(mtow / 50) * base_rate_map[base]
        for fir in ("CZQMFIR", "CZULFIR"):
            if fir not in df.columns:
                continue
            dist_km = df[fir].fillna(0) * 1.852
            total += NAV_CANADA_R * np.sqrt(mtow) * dist_km * CAD_EUR
        if "CZQXFIR" in df.columns:
            total += (df["CZQXFIR"].fillna(0) > 0).astype(float) * NAV_CANADA_OCEANIC * CAD_EUR
        return total.where(mtow.notna()).round(2)

    def _fuel_vec(df):
        fuel_kgh = df[ac_col].map(FUEL_KGH)
        return (fuel_kgh * df["duration_h"] * JET_A_EUR_PER_KG).where(
            fuel_kgh.notna() & df["mtow_t"].notna()
        ).round(2)

    df_full["atc_eur"]  = _atc_vec(df_full)
    df_full["fuel_eur"] = _fuel_vec(df_full)
    df_full["cost_eur"] = (df_full["atc_eur"] + df_full["fuel_eur"]).round(2)
    dist_firs = [c for c in rate_firs if c.endswith("FIR") and c != "CZQXFIR"]
    df_full["total_dist_nm"] = df_full[dist_firs].fillna(0).sum(axis=1)

    agg_dict = {
        "n_flights":          ("ECTRL ID", "count"),
        "mean_duration_h":    ("duration_h", "mean"),
        "most_common_ac":     (ac_col, lambda x: x.mode().iloc[0]),
        "mean_total_dist_nm": ("total_dist_nm", "mean"),
    }
    agg_dict.update({f"mean_{f}": (f, "mean") for f in rate_firs})
    full_summary = df_full.groupby(["ADEP", "ADES", "cluster_kmeans"]).agg(**agg_dict).reset_index()

    cost_agg = (
        df_full.dropna(subset=["cost_eur"])
        .groupby(["ADEP", "ADES", "cluster_kmeans"])["cost_eur"]
        .agg(mean_cost_eur="mean", std_cost_eur="std").round(2).reset_index()
    )
    full_summary = full_summary.merge(cost_agg, on=["ADEP", "ADES", "cluster_kmeans"], how="left")
    full_summary["n_clusters_od"] = full_summary.groupby(["ADEP", "ADES"])["cluster_kmeans"].transform("nunique")

    ac_agg_dict = {
        "n_flights_ac":          ("ECTRL ID", "count"),
        "mean_duration_h_ac":    ("duration_h", "mean"),
        "mean_total_dist_nm_ac": ("total_dist_nm", "mean"),
    }
    ac_agg_dict.update({f"mean_{f}_ac": (f, "mean") for f in rate_firs})
    full_summary_ac = (
        df_full.groupby(["ADEP", "ADES", "cluster_kmeans", ac_col])
        .agg(**ac_agg_dict)
        .reset_index()
        .rename(columns={ac_col: "ac_type"})
    )

    return full_summary, full_summary_ac


def predict_route_options(ac_type, adep, ades, full_summary, full_summary_ac,
                           sort_by="cost_eur", min_n_ac=MIN_N_AC):
    """Ranked route alternatives for ac_type on adep-ades, from the full-dataset summary."""
    od_rows = full_summary[(full_summary["ADEP"] == adep) & (full_summary["ADES"] == ades)]
    if od_rows.empty:
        raise ValueError(f"No clusters found for {adep}-{ades}")

    mtow = MTOW_TONNES.get(ac_type)
    if mtow is None:
        raise ValueError(f"'{ac_type}' not in MTOW_TONNES")
    fuel_kgh = FUEL_KGH.get(ac_type)
    if fuel_kgh is None:
        raise ValueError(f"'{ac_type}' not in FUEL_KGH")

    ac_rows = full_summary_ac[
        (full_summary_ac["ADEP"] == adep) & (full_summary_ac["ADES"] == ades)
        & (full_summary_ac["ac_type"] == ac_type)
    ].set_index("cluster_kmeans")

    rate_firs = list(EUROCONTROL_RATES) + ["CZQMFIR", "CZULFIR", "CZQXFIR"]
    results = []
    for _, row in od_rows.iterrows():
        cluster = row["cluster_kmeans"]
        ac_row = ac_rows.loc[cluster] if cluster in ac_rows.index else None
        ac_specific = ac_row is not None and ac_row["n_flights_ac"] >= min_n_ac

        if ac_specific:
            duration = ac_row["mean_duration_h_ac"]
            rep_row  = {fir: ac_row.get(f"mean_{fir}_ac", 0) for fir in rate_firs}
            dist_nm  = ac_row.get("mean_total_dist_nm_ac")
            n_hist   = int(ac_row["n_flights_ac"])
        else:
            duration = row["mean_duration_h"]
            rep_row  = {fir: row.get(f"mean_{fir}", 0) for fir in rate_firs}
            dist_nm  = row.get("mean_total_dist_nm")
            n_hist   = int(row["n_flights"])

        rep_row["mtow_t"] = mtow
        pred_atc  = round(flight_atc_eur(rep_row), 2)
        pred_fuel = round(fuel_kgh * duration * JET_A_EUR_PER_KG, 2)

        results.append({
            "cluster":              cluster,
            "ac_specific":          ac_specific,
            "n_historical_flights": n_hist,
            "mean_dist_nm":         round(float(dist_nm), 1) if pd.notna(dist_nm) else None,
            "mean_duration_h":      round(duration, 3),
            "predicted_atc_eur":    pred_atc,
            "predicted_fuel_eur":   pred_fuel,
            "predicted_cost_eur":   round(pred_atc + pred_fuel, 2),
        })

    sort_col = "predicted_cost_eur" if sort_by == "cost_eur" else "mean_duration_h"
    df_out = pd.DataFrame(results).sort_values(sort_col).reset_index(drop=True)
    df_out.index += 1
    df_out.index.name = "rank"
    return df_out


def query_route_profile(ac_type, adep, ades, l3_centroids, l3_centroids_pooled,
                         error_summary, error_summary_pooled, ac_col,
                         sort_by="cost_eur", min_n_ac=MIN_N_AC):
    """Like predict_route_options, but scoped to the training/validation sample and
    including the Type 1 error envelope (delta_*_mean/std/p5/p95) per cluster.

    Falls back to the pooled-across-aircraft-type cluster centroid (with
    fuel/cost recomputed via formula for the queried aircraft) when ac_type
    has fewer than min_n_ac historical flights in a cluster - a blended
    historical fuel/cost across aircraft types isn't meaningful.
    """
    pooled = l3_centroids_pooled[(l3_centroids_pooled["ADEP"] == adep) & (l3_centroids_pooled["ADES"] == ades)]
    if pooled.empty:
        raise ValueError(
            f"No L3 data for {adep}-{ades}. Run the clustering pipeline for this OD pair first, "
            f"or use predict_route_options for the full dataset."
        )

    mtow = MTOW_TONNES.get(ac_type)
    if mtow is None:
        raise ValueError(f"'{ac_type}' not in MTOW_TONNES")
    fuel_kgh = FUEL_KGH.get(ac_type)
    if fuel_kgh is None:
        raise ValueError(f"'{ac_type}' not in FUEL_KGH")

    ac_cen = l3_centroids[
        (l3_centroids["ADEP"] == adep) & (l3_centroids["ADES"] == ades) & (l3_centroids[ac_col] == ac_type)
    ].set_index("cluster_kmeans")

    delta_cols = [c for c in error_summary.columns if c.startswith("delta_")]

    ac_err = error_summary[
        (error_summary["ADEP"] == adep) & (error_summary["ADES"] == ades) & (error_summary[ac_col] == ac_type)
    ].set_index("cluster_kmeans")

    pooled_err = error_summary_pooled[
        (error_summary_pooled["ADEP"] == adep) & (error_summary_pooled["ADES"] == ades)
    ].set_index("cluster_kmeans")

    rate_firs = rate_firs_in(pooled)
    rows = []
    for _, prow in pooled.iterrows():
        cluster = prow["cluster_kmeans"]
        ac_row = ac_cen.loc[cluster] if cluster in ac_cen.index else None
        ac_specific = ac_row is not None and ac_row["n_l3"] >= min_n_ac

        if ac_specific:
            n_l3     = int(ac_row["n_l3"])
            dist_nm  = ac_row["centroid_dist_nm"]
            duration = ac_row["centroid_duration_h"]
            fuel_kg  = ac_row["centroid_fuel_kg"]
            cost_eur = ac_row["centroid_cost_eur"]
            err_row  = ac_err.loc[cluster] if cluster in ac_err.index else None
        else:
            n_l3     = int(prow["n_l3"])
            dist_nm  = prow["centroid_dist_nm"]
            duration = prow["centroid_duration_h"]
            rep_row  = {fir: prow.get(f"mean_{fir}", 0) for fir in rate_firs}
            rep_row["mtow_t"] = mtow
            fuel_kg  = round(fuel_kgh * duration, 1)
            cost_eur = round(flight_atc_eur(rep_row) + fuel_kg * JET_A_EUR_PER_KG, 2)
            err_row  = pooled_err.loc[cluster] if cluster in pooled_err.index else None

        row = {
            "cluster_kmeans": cluster, "ac_specific": ac_specific, "n_l3": n_l3,
            "centroid_dist_nm": dist_nm, "centroid_duration_h": duration,
            "centroid_fuel_kg": fuel_kg, "centroid_cost_eur": cost_eur,
        }
        for c in delta_cols:
            if not ac_specific and c.startswith("delta_fuel_kg"):
                row[c] = np.nan
            elif err_row is not None and c in err_row.index:
                row[c] = err_row[c]
            else:
                row[c] = np.nan
        rows.append(row)

    merged = pd.DataFrame(rows)
    merged["reliable"] = merged["n_l3"] >= 10

    sort_map = {"cost_eur": "centroid_cost_eur", "duration_h": "centroid_duration_h", "dist_nm": "centroid_dist_nm"}
    sort_col = sort_map.get(sort_by, sort_by)

    out_cols = [
        "cluster_kmeans", "ac_specific", "n_l3", "reliable",
        "centroid_dist_nm", "delta_dist_nm_mean", "delta_dist_nm_std", "delta_dist_nm_p5", "delta_dist_nm_p95",
        "centroid_duration_h", "delta_duration_h_mean", "delta_duration_h_std", "delta_duration_h_p5", "delta_duration_h_p95",
        "centroid_fuel_kg", "delta_fuel_kg_mean", "delta_fuel_kg_std", "delta_fuel_kg_p5", "delta_fuel_kg_p95",
        "centroid_cost_eur",
    ]
    df_out = merged[[c for c in out_cols if c in merged.columns]].sort_values(sort_col).reset_index(drop=True)
    df_out.index += 1
    df_out.index.name = "rank"
    return df_out
