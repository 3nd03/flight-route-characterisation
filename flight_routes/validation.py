"""Cluster quality checks and outlier-robust error summaries.

OOD validation itself isn't a separate function here - it's the same
data-loading, clustering, cost and variance functions from the other
modules, just run on a held-out O-D pair and compared against the training
result. cluster_quality() and summarise_with_outlier_removal() are the
genuinely separate validation logic.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import pairwise_distances

from .features import make_route_signatures


def cluster_quality(df, fir_cols, label_col="cluster_kmeans"):
    """Per O-D pair: k, silhouette (Hamming), mean between/within-cluster Hamming distance.

    A low silhouette score with a much smaller within- than between-cluster
    distance means clusters are tight relative to their separation despite
    the score - typically because several clusters share an identical modal
    FIR signature and differ only in within-corridor distance, which the
    binary-Hamming metric alone doesn't reward.
    """
    sigs = make_route_signatures(df, fir_cols)
    rows = []
    for (adep, ades), grp in df.groupby(["ADEP", "ADES"]):
        X = sigs.loc[grp.index].values.astype(float)
        labels = grp[label_col].values
        k = len(set(labels))

        if k == 1:
            rows.append({"ADEP": adep, "ADES": ades, "k": k,
                         "silhouette": np.nan, "between_hamming": np.nan, "within_hamming": np.nan})
            continue

        sil = silhouette_score(X, labels, metric="hamming")

        centroids = np.array([X[labels == c].mean(axis=0) for c in sorted(set(labels))])
        b_dist = pairwise_distances(centroids, metric="hamming")
        np.fill_diagonal(b_dist, np.nan)
        mean_between = np.nanmean(b_dist)

        within_vals = [
            pairwise_distances(X[labels == c], metric="hamming").mean()
            for c in sorted(set(labels)) if (labels == c).sum() > 1
        ]
        mean_within = np.mean(within_vals) if within_vals else np.nan

        rows.append({"ADEP": adep, "ADES": ades, "k": k,
                     "silhouette": sil, "between_hamming": mean_between, "within_hamming": mean_within})
    return pd.DataFrame(rows)


def mad_filter(vals, threshold=3.5):
    """Modified z-score (Iglewicz & Hoaglin) outlier filter - robust at small n.

    Returns (clean_values, n_removed).
    """
    arr = np.asarray(vals, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return arr, 0
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    if mad == 0:
        return arr, 0
    mod_z = 0.6745 * np.abs(arr - med) / mad
    mask = mod_z <= threshold
    return arr[mask], int((~mask).sum())


def summarise_with_outlier_removal(df, group_cols, delta_cols, threshold=3.5, strip_suffix=None):
    """Like the plain per-group mean/std/p5/p95 summary, but MAD-filters outliers within
    each group first. Pass strip_suffix (e.g. '_pooled') to drop a column-name suffix from
    the output, so pooled and non-pooled summaries share the same column names.
    """
    rows = []
    for key, grp in df.dropna(subset=delta_cols).groupby(group_cols):
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key_tuple))
        row["n"] = len(grp)
        total_removed = 0
        for col in delta_cols:
            clean, n_rem = mad_filter(grp[col].values, threshold)
            total_removed += n_rem
            out_col = col[: -len(strip_suffix)] if strip_suffix and col.endswith(strip_suffix) else col
            if len(clean) >= 2:
                row[f"{out_col}_mean"] = round(float(np.mean(clean)), 2)
                row[f"{out_col}_std"]  = round(float(np.std(clean, ddof=1)), 2)
                row[f"{out_col}_p5"]   = round(float(np.percentile(clean, 5)), 2)
                row[f"{out_col}_p95"]  = round(float(np.percentile(clean, 95)), 2)
            else:
                for sfx in ("_mean", "_std", "_p5", "_p95"):
                    row[f"{out_col}{sfx}"] = np.nan
        row["n_outliers"] = total_removed
        rows.append(row)
    return pd.DataFrame(rows)
