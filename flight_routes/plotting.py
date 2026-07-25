"""All Plotly visualisations: FIR usage heatmaps, cluster separation (PCA),
cost-vs-duration scatter, planned-vs-realised noise histograms, representative
trajectories, and the full-dataset coverage map. Folds in what used to be the
standalone notebooks/plot_routes.py.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_fir_heatmap(df, fir_cols, label_col="cluster_kmeans"):
    """One heatmap per O-D pair: mean distance per FIR, one row per cluster."""
    for (adep, ades), grp in df.groupby(["ADEP", "ADES"]):
        clusters = sorted(grp[label_col].unique())
        pair_firs = [f for f in fir_cols if (grp[f] > 0).mean() >= 0.10]
        if not pair_firs:
            continue

        n_list = [grp[grp[label_col] == c].shape[0] for c in clusters]
        z = [[round(grp[grp[label_col] == c][f].mean(), 1) for f in pair_firs] for c in clusters]
        y_labels = [f"C{c}  (n={n_list[i]})" for i, c in enumerate(clusters)]
        x_labels = [f.replace("FIR", "").replace("UIR", "") for f in pair_firs]

        fig = go.Figure(go.Heatmap(
            z=z, x=x_labels, y=y_labels,
            colorscale=[[0, "green"], [0.5, "yellow"], [1, "red"]], colorbar_title="mean nm",
        ))
        fig.update_layout(
            title=f"{adep}-{ades}: FIR usage by cluster (mean distance, nm)",
            xaxis_tickangle=-45,
            height=max(300, len(clusters) * 55 + 150),
            margin=dict(l=130, b=130),
        )
        fig.show()


def plot_cluster_pca(df, fir_cols, label_col="cluster_kmeans"):
    """PCA of binary FIR signatures, coloured by cluster - visual check of cluster separation."""
    from sklearn.decomposition import PCA

    sigs = (df[fir_cols] > 0).astype(int)
    for (adep, ades), grp in df.groupby(["ADEP", "ADES"]):
        if grp[label_col].nunique() < 2:
            continue

        X = sigs.loc[grp.index].values
        coords = PCA(n_components=2, random_state=158).fit_transform(X)
        plot_df = pd.DataFrame({
            "PC1": coords[:, 0], "PC2": coords[:, 1],
            "cluster": grp[label_col].astype(str).values,
        })
        fig = px.scatter(
            plot_df, x="PC1", y="PC2", color="cluster",
            title=f"{adep}-{ades}: cluster separation (PCA of binary FIR signatures)",
            labels={"cluster": "Cluster"},
        )
        fig.update_traces(marker_size=6, marker_opacity=0.7)
        fig.update_layout(height=500)
        fig.show()


def plot_cost_bars(df, label_col="cluster_kmeans"):
    """Stacked ATC/fuel cost bar per cluster, with total and n annotated."""
    for (adep, ades), grp in df.groupby(["ADEP", "ADES"]):
        grp = grp.dropna(subset=["atc_eur", "fuel_eur"])
        clusters = sorted(grp[label_col].unique())
        x = [f"C{c}" for c in clusters]
        atc_m  = [round(grp[grp[label_col] == c]["atc_eur"].mean()) for c in clusters]
        fuel_m = [round(grp[grp[label_col] == c]["fuel_eur"].mean()) for c in clusters]
        n_list = [grp[grp[label_col] == c].shape[0] for c in clusters]

        fig = go.Figure(data=[
            go.Bar(name="ATC", x=x, y=atc_m, marker_color="#4C78A8"),
            go.Bar(name="Fuel", x=x, y=fuel_m, marker_color="#F58518"),
        ])
        for xi, atc, fuel, n in zip(x, atc_m, fuel_m, n_list):
            fig.add_annotation(x=xi, y=atc + fuel, text=f"€{atc + fuel:,}<br>n={n}",
                                showarrow=False, yanchor="bottom", font_size=11)
        fig.update_layout(barmode="stack", title=f"{adep}-{ades}: mean cost per cluster",
                           yaxis_title="EUR", height=460)
        fig.show()


def plot_route_alternatives(df_summary, title_suffix=""):
    """Cost vs duration scatter, one point per cluster, sized by n_flights."""
    for (adep, ades), grp in df_summary.groupby(["ADEP", "ADES"]):
        if grp["mean_cost_eur"].isna().all():
            continue
        fig = px.scatter(
            grp, x="mean_duration_h", y="mean_cost_eur", size="n_flights",
            color=grp["cluster_kmeans"].astype(str), text=grp["cluster_kmeans"].astype(str),
            title=f"{adep}-{ades}: route alternatives{title_suffix}",
            labels={"mean_duration_h": "Mean duration (h)", "mean_cost_eur": "Mean cost (€)", "color": "Cluster"},
            size_max=40,
        )
        fig.update_traces(textposition="top center", marker_opacity=0.85)
        fig.update_layout(height=520, showlegend=False)
        fig.show()


def plot_noise_distribution(df, delta_cols, ac_col=None, label_col="cluster_kmeans"):
    """Histogram grid (rows = cluster, cols = delta_cols) of planned-vs-realised noise,
    coloured by aircraft type. Works for either comparison - pass variance.DELTA_COLS
    (vs cluster centroid) or variance.DELTA_COLS_SELF (vs own filed plan).
    """
    if ac_col is None:
        ac_col = next(c for c in df.columns if "AC Type" in c)
    df = df.dropna(subset=delta_cols)

    for (adep, ades), grp in df.groupby(["ADEP", "ADES"]):
        clusters = sorted(grp[label_col].unique())
        ac_types = sorted(grp[ac_col].unique())
        ac_colours = {ac: px.colors.qualitative.Plotly[i % 10] for i, ac in enumerate(ac_types)}

        fig = make_subplots(
            rows=len(clusters), cols=len(delta_cols),
            subplot_titles=[f"C{c} | {col}" for c in clusters for col in delta_cols],
        )
        seen_ac = set()
        for r, cluster in enumerate(clusters, start=1):
            cluster_grp = grp[grp[label_col] == cluster]
            for c_idx, col in enumerate(delta_cols, start=1):
                for ac in sorted(cluster_grp[ac_col].unique()):
                    ac_vals = cluster_grp.loc[cluster_grp[ac_col] == ac, col].dropna()
                    if ac_vals.empty:
                        continue
                    fig.add_trace(
                        go.Histogram(x=ac_vals, name=ac, nbinsx=20, marker_color=ac_colours[ac],
                                     legendgroup=ac, showlegend=(ac not in seen_ac)),
                        row=r, col=c_idx,
                    )
                    seen_ac.add(ac)
                fig.add_vline(x=0, line_dash="dash", line_color="#E45756", row=r, col=c_idx)

        fig.update_layout(
            title_text=f"{adep}-{ades}: noise (actual - reference) by cluster and AC type",
            height=max(300, len(clusters) * 220), barmode="overlay", margin=dict(t=100),
        )
        fig.update_traces(opacity=0.65)
        fig.show()


def representative_flight_id(grp_df, centroid_row):
    """ECTRL ID of the real flight in grp_df closest to centroid_row on planned dist/duration/fuel/cost."""
    feats = ["planned_dist_nm", "planned_duration_h", "planned_fuel_kg", "cost_eur"]
    cent = np.array([centroid_row["centroid_dist_nm"], centroid_row["centroid_duration_h"],
                      centroid_row["centroid_fuel_kg"], centroid_row["centroid_cost_eur"]])
    sub = grp_df[feats].dropna()
    if sub.empty:
        return None
    diffs = sub.values - cent
    scale = np.abs(diffs).max(axis=0) + 1e-9
    return grp_df.loc[sub.iloc[np.linalg.norm(diffs / scale, axis=1).argmin()].name, "ECTRL ID"]


def plot_representative_trajectories(df_compared, l3_centroids, actual_pts, ac_col=None):
    """Actual trajectory of the representative flight, one figure per (O-D pair, cluster)."""
    ac_col = ac_col or next(c for c in df_compared.columns if "AC Type" in c)

    for (adep, ades), od_grp in df_compared.groupby(["ADEP", "ADES"]):
        for clust in sorted(od_grp["cluster_kmeans"].unique()):
            c_grp = od_grp[od_grp["cluster_kmeans"] == clust]
            top_ac = c_grp[ac_col].value_counts().idxmax()
            cen = l3_centroids[
                (l3_centroids["ADEP"] == adep) & (l3_centroids["ADES"] == ades)
                & (l3_centroids["cluster_kmeans"] == clust) & (l3_centroids[ac_col] == top_ac)
            ]
            if cen.empty:
                continue
            rep_id = representative_flight_id(c_grp[c_grp[ac_col] == top_ac], cen.iloc[0])
            if rep_id is None:
                continue

            pts = actual_pts[actual_pts["ECTRL ID"] == str(rep_id)].copy()
            pts["time_dt"] = pd.to_datetime(pts["Time Over"], dayfirst=True)
            pts = pts.sort_values("time_dt")
            if pts.empty:
                continue

            dr = df_compared[df_compared["ECTRL ID"] == rep_id]
            d_dist = dr["delta_dist_nm"].values[0] if not dr.empty else np.nan
            d_dur  = dr["delta_duration_h"].values[0] if not dr.empty else np.nan
            d_dist_str = f"{d_dist:+.0f} nm" if not pd.isna(d_dist) else "N/A"
            d_dur_str  = f"{d_dur:+.3f} h" if not pd.isna(d_dur) else "N/A"

            fig = go.Figure(go.Scattergeo(
                lat=pts["Latitude"].astype(float), lon=pts["Longitude"].astype(float),
                mode="lines+markers", line=dict(width=2, color="#4C78A8"), marker=dict(size=3),
            ))
            fig.update_geos(
                projection_type="natural earth", fitbounds="locations",
                showland=True, landcolor="#E8E8E8", showocean=True, oceancolor="#C9DEF4",
                showcoastlines=True, coastlinecolor="#999999",
            )
            fig.update_layout(
                title=(f"{adep}-{ades} | Cluster {clust} representative actual trajectory<br>"
                       f"ECTRL {rep_id} ({top_ac}), Δdist {d_dist_str} | Δdur {d_dur_str}"),
                height=450, margin=dict(t=80),
            )
            fig.show()


def plot_coverage_map(full_summary):
    """World map of every O-D pair, coloured by how many route alternatives it has.

    Requires the `airportsdata` package (pip install airportsdata) for airport
    coordinates - not a hard dependency of the library, only of this one plot.
    """
    try:
        import airportsdata
    except ImportError as e:
        raise ImportError("plot_coverage_map needs `pip install airportsdata`") from e

    airports_db = airportsdata.load("ICAO")
    lat_map = {k: v["lat"] for k, v in airports_db.items()}
    lon_map = {k: v["lon"] for k, v in airports_db.items()}

    od_map = full_summary[["ADEP", "ADES", "n_clusters_od"]].drop_duplicates().copy()
    od_map["adep_lat"] = od_map["ADEP"].map(lat_map)
    od_map["adep_lon"] = od_map["ADEP"].map(lon_map)
    od_map["ades_lat"] = od_map["ADES"].map(lat_map)
    od_map["ades_lon"] = od_map["ADES"].map(lon_map)
    od_map = od_map.dropna(subset=["adep_lat", "ades_lat"]).reset_index(drop=True)

    def _arc_coords(subset):
        lats, lons = [], []
        for _, r in subset.iterrows():
            lats += [r["adep_lat"], r["ades_lat"], None]
            lons += [r["adep_lon"], r["ades_lon"], None]
        return lats, lons

    groups = [
        (od_map[od_map["n_clusters_od"] == 1], "k=1 (single route)", "#7799BB", 0.12, 0.7),
        (od_map[od_map["n_clusters_od"] == 2], "k=2 (two variants)", "#54A0E0", 0.35, 1.2),
        (od_map[od_map["n_clusters_od"] >= 3], "k>=3 (three+ variants)", "#F58518", 0.48, 1.4),
    ]

    fig = go.Figure()
    for subset, name, colour, opacity, width in groups:
        lats, lons = _arc_coords(subset)
        fig.add_trace(go.Scattergeo(
            lat=lats, lon=lons, mode="lines", line=dict(width=width, color=colour),
            opacity=opacity, name=f"{name} -- {len(subset):,} pairs", hoverinfo="skip",
        ))

    n_alt = od_map[od_map["n_clusters_od"] >= 2].shape[0]
    fig.update_geos(
        projection_type="natural earth", showland=True, landcolor="#1a1a2e",
        showocean=True, oceancolor="#16213e", showcoastlines=True, coastlinecolor="#444466",
        showframe=False, lataxis_range=[-60, 80],
    )
    fig.update_layout(
        title=dict(
            text=(f"Flight route coverage<br><sup>{len(od_map):,} O-D pairs | "
                  f"{n_alt:,} with route alternatives ({n_alt / len(od_map) * 100:.1f}%)</sup>"),
            x=0.5, xanchor="center", font=dict(color="white"),
        ),
        legend=dict(x=1.0, y=0.0, xanchor="right", yanchor="bottom", bgcolor="rgba(15,15,35,0.7)", font=dict(color="white")),
        paper_bgcolor="#0f0f23", height=560, margin=dict(t=80, b=10, l=10, r=10),
    )
    fig.show()


# --- mean-representative-route map (formerly the standalone plot_routes.py) ---

PALETTE = px.colors.qualitative.Set1 + px.colors.qualitative.Set2 + px.colors.qualitative.Set3
N_INTERP = 100
SEPARATION_THRESHOLD_KM = 50.0


def mean_route(cluster_pts, n_interp=N_INTERP):
    """Normalise each flight in a cluster to n_interp points, then average lat/lon across flights."""
    from scipy.interpolate import interp1d

    lats, lons = [], []
    for _, flight in cluster_pts.groupby("ECTRL_ID", sort=False):
        flight = flight.sort_values("Sequence Number")
        if len(flight) < 2:
            continue
        x = np.linspace(0, 1, len(flight))
        xi = np.linspace(0, 1, n_interp)
        lats.append(interp1d(x, flight["Latitude"].values)(xi))
        lons.append(interp1d(x, flight["Longitude"].values)(xi))
    if not lats:
        return None, None
    return np.mean(lats, axis=0), np.mean(lons, axis=0)


def _haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def separation_report(adep, ades, clusters, mean_lats, mean_lons, threshold_km=SEPARATION_THRESHOLD_KM):
    """Pairwise max lateral deviation (km) between cluster mean routes - flags pairs that
    look like the same route (below threshold) despite being labelled as different clusters.
    """
    n = len(clusters)
    if n < 2:
        return {"status": "single cluster, no separation to evaluate"}

    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if mean_lats[i] is None or mean_lats[j] is None:
                continue
            max_dev = _haversine_km(mean_lats[i], mean_lons[i], mean_lats[j], mean_lons[j]).max()
            pairs.append({
                "cluster_a": clusters[i], "cluster_b": clusters[j],
                "max_deviation_km": round(float(max_dev), 1),
                "below_threshold": max_dev < threshold_km,
            })
    return pairs


def plot_od_representative_routes(od_pts, adep, ades):
    """One map per O-D pair: the mean representative route line for each cluster."""
    clusters = sorted(od_pts["cluster_kmeans"].unique())
    n = od_pts["ECTRL_ID"].nunique()

    lat_min, lat_max = od_pts["Latitude"].min(), od_pts["Latitude"].max()
    lon_min, lon_max = od_pts["Longitude"].min(), od_pts["Longitude"].max()
    lat_pad = (lat_max - lat_min) * 0.1 + 3
    lon_pad = (lon_max - lon_min) * 0.1 + 3

    fig = go.Figure()
    all_mean_lats, all_mean_lons = [], []
    for i, c in enumerate(clusters):
        c_pts = od_pts[od_pts["cluster_kmeans"] == c]
        n_clust = c_pts["ECTRL_ID"].nunique()
        mean_lat, mean_lon = mean_route(c_pts)
        all_mean_lats.append(mean_lat)
        all_mean_lons.append(mean_lon)
        if mean_lat is not None:
            fig.add_trace(go.Scattergeo(
                lon=mean_lon, lat=mean_lat, mode="lines",
                name=f"Cluster {c}  (n={n_clust})",
                line=dict(width=3, color=PALETTE[i % len(PALETTE)]),
            ))

    separation = separation_report(adep, ades, clusters, all_mean_lats, all_mean_lons)

    fig.update_geos(
        projection_type="natural earth",
        lataxis_range=[lat_min - lat_pad, lat_max + lat_pad],
        lonaxis_range=[lon_min - lon_pad, lon_max + lon_pad],
        showland=True, landcolor="rgb(230,230,230)",
        showocean=True, oceancolor="rgb(200,215,235)",
        showcoastlines=True, coastlinecolor="white",
        showcountries=True, countrycolor="white",
        showframe=False,
    )
    fig.update_layout(
        title=dict(text=f"{adep} -> {ades}  |  {n} flights  |  {len(clusters)} clusters", x=0.5),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)", bordercolor="lightgrey", borderwidth=1),
        margin=dict(l=0, r=0, t=50, b=0),
        height=600,
    )
    fig.show()
    return separation
