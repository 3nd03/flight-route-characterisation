"""Three-layer route clustering.

L1: O-D pair (groupby).
L1.5: hard split by binary FIR-crossing signature (exact grouping, not KMeans) -
      guarantees flights that crossed genuinely different named airspace are
      never merged by the distance-based step that follows.
L2: KMeans on standardised FIR distances within each L1.5 signature group,
    k chosen by silhouette score.
Then any cluster below MIN_CLUSTER_SIZE is force-merged into its nearest
neighbour by centroid distance (merge_similar_clusters), regardless of the
normal similarity threshold - this closes the gap where a hard L1.5 split
can produce a 1-4 flight cluster that KMeans' own size guard never sees.
"""

import gc
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .costs import _parse_duration
from .features import active_firs_for_group

MIN_CLUSTER_SIZE    = 15
MERGE_THRESHOLD_NM   = 100.0
RANDOM_STATE         = 158


def _best_k(X, k_range=range(2, 6), min_cluster_size=MIN_CLUSTER_SIZE):
    """Pick k by silhouette score, rejecting any k that produces a cluster below min_cluster_size."""
    scores = {}
    for k in k_range:
        if len(X) <= k:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            labels = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit_predict(X)
        if len(set(labels)) < k:
            continue
        if pd.Series(labels).value_counts().min() < min_cluster_size:
            continue
        scores[k] = silhouette_score(X, labels)
    return max(scores, key=scores.get) if scores else 1


def cluster_od_3layer(df, fir_cols):
    """Cluster every O-D pair in df. Returns (labels_l15, labels_final)."""
    labels_l15   = pd.Series(-1, index=df.index, dtype=int)
    labels_final = pd.Series(-1, index=df.index, dtype=int)

    for (adep, ades), od_grp in df.groupby(["ADEP", "ADES"]):
        sig_series = (od_grp[fir_cols] > 0).astype(int).apply(tuple, axis=1)

        cluster_counter = 0
        sig_counter = 0
        for sig_val, sig_grp in od_grp.groupby(sig_series):
            labels_l15.loc[sig_grp.index] = sig_counter
            sig_counter += 1

            sub_X = StandardScaler().fit_transform(sig_grp[fir_cols].fillna(0).values)
            k2 = _best_k(sub_X)
            l2 = (
                np.zeros(len(sig_grp), dtype=int) if k2 == 1
                else KMeans(n_clusters=k2, random_state=RANDOM_STATE, n_init=10).fit_predict(sub_X)
            )

            for l2_id in sorted(set(l2)):
                labels_final.loc[sig_grp.index[l2 == l2_id]] = cluster_counter
                cluster_counter += 1

    return labels_l15, labels_final


def merge_similar_clusters(df, fir_cols, label_col="cluster_kmeans",
                            threshold=MERGE_THRESHOLD_NM, min_size=MIN_CLUSTER_SIZE):
    """Collapse clusters with near-identical FIR-distance profiles; force-merge undersized ones."""
    df = df.copy()
    changed = True
    while changed:
        changed = False
        for (adep, ades), grp in df.groupby(["ADEP", "ADES"]):
            clusters = sorted(grp[label_col].unique())
            if len(clusters) < 2:
                continue
            centroids = {c: grp.loc[grp[label_col] == c, fir_cols].mean().values for c in clusters}
            sizes     = {c: int((grp[label_col] == c).sum()) for c in clusters}

            undersized = [c for c in clusters if sizes[c] < min_size]
            if undersized:
                src = min(undersized, key=lambda c: sizes[c])
                dst = min((c for c in clusters if c != src),
                          key=lambda c: np.linalg.norm(centroids[src] - centroids[c]))
                df.loc[(df["ADEP"] == adep) & (df["ADES"] == ades) & (df[label_col] == src), label_col] = dst
                changed = True
                break

            min_dist, a, b = np.inf, None, None
            for i, ci in enumerate(clusters):
                for cj in clusters[i + 1:]:
                    d = np.linalg.norm(centroids[ci] - centroids[cj])
                    if d < min_dist:
                        min_dist, a, b = d, ci, cj
            if min_dist < threshold:
                na, nb = sizes[a], sizes[b]
                src, dst = (a, b) if na < nb else (b, a)
                df.loc[(df["ADEP"] == adep) & (df["ADES"] == ades) & (df[label_col] == src), label_col] = dst
                changed = True
                break

    for (adep, ades), grp in df.groupby(["ADEP", "ADES"]):
        remap = {old: new for new, old in enumerate(sorted(grp[label_col].unique()))}
        df.loc[(df["ADEP"] == adep) & (df["ADES"] == ades), label_col] = grp[label_col].map(remap)
    return df


def cluster_summary(df, fir_cols, label_col="cluster_kmeans"):
    """Per-cluster stats: flight count, mean duration, most common aircraft, mean FIR distances."""
    df = df.copy()
    df["duration_h"] = df["Duration_Hours"].apply(_parse_duration)
    ac_col = next(c for c in df.columns if "AC Type" in c)

    agg = {
        "n_flights":       ("ECTRL ID", "count"),
        "mean_duration_h": ("duration_h", "mean"),
        "most_common_ac":  (ac_col, lambda x: x.mode().iloc[0]),
    }
    agg.update({f"mean_{f}": (f, "mean") for f in fir_cols})
    return df.groupby(["ADEP", "ADES", label_col]).agg(**agg).reset_index()


def cluster_full_dataset(df, all_fir_cols, progress_every=500):
    """Cluster every qualifying O-D pair in the full dataset. Returns (labels, diagnostics).

    Streams the groupby instead of materialising every sub-frame up front, and
    runs gc.collect() periodically - holding all O-D pair copies in memory
    simultaneously (~15k KMeans fits total) is what OOM'd this on Colab's
    free-tier RAM at ~6500/7345 pairs in the original notebook run.
    """
    labels_final = pd.Series(-1, index=df.index, dtype=int)
    diagnostics  = []
    n = df.groupby(["ADEP", "ADES"]).ngroups

    for i, ((adep, ades), od_grp) in enumerate(df.groupby(["ADEP", "ADES"])):
        if (i + 1) % progress_every == 0 or i == n - 1:
            gc.collect()

        pair_firs = active_firs_for_group(od_grp, all_fir_cols)
        if not pair_firs:
            labels_final.loc[od_grp.index] = 0
            diagnostics.append({"ADEP": adep, "ADES": ades,
                                 "n_flights": len(od_grp), "n_clusters": 1})
            continue

        bin_X = (od_grp[pair_firs] > 0).astype(int).values
        k2 = _best_k(bin_X)
        l2 = (
            np.zeros(len(od_grp), dtype=int) if k2 == 1
            else KMeans(n_clusters=k2, random_state=RANDOM_STATE, n_init=3).fit_predict(bin_X)
        )

        cluster_counter = 0
        for l2_id in sorted(set(l2)):
            mask = l2 == l2_id
            sub_idx = od_grp.index[mask]
            sub_X = StandardScaler().fit_transform(od_grp.loc[sub_idx, pair_firs].fillna(0).values)
            k3 = _best_k(sub_X)
            l3 = (
                np.zeros(mask.sum(), dtype=int) if k3 == 1
                else KMeans(n_clusters=k3, random_state=RANDOM_STATE, n_init=3).fit_predict(sub_X)
            )
            for l3_id in sorted(set(l3)):
                labels_final.loc[sub_idx[l3 == l3_id]] = cluster_counter
                cluster_counter += 1

        diagnostics.append({"ADEP": adep, "ADES": ades,
                             "n_flights": len(od_grp), "n_clusters": cluster_counter})

    return labels_final, pd.DataFrame(diagnostics)
