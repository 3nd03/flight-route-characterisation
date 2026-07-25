"""Local viewer: mean representative route per cluster, one map per O-D pair.
Logic lives in flight_routes.plotting (mean_route, separation_report,
plot_od_representative_routes) - this script just loads the local CSVs and calls it.
"""

import sys

import pandas as pd

sys.path.insert(0, r"C:\Users\User\Desktop\Projects\DC TRL files\Analysis\DatasetsF\flight-route-characterisation")
from flight_routes.plotting import plot_od_representative_routes

FLIGHT_POINTS  = r"C:\Users\User\Desktop\Projects\DC TRL files\Analysis\Flight_Points_Filed_20230901_20230930.csv"
CLUSTER_LABELS = r"C:\Users\User\Desktop\Projects\DC TRL files\Analysis\DatasetsF\flight-route-characterisation\data\processed\cluster_labels.csv"

pts    = pd.read_csv(FLIGHT_POINTS)
labels = pd.read_csv(CLUSTER_LABELS)

pts.columns = pts.columns.str.strip().str.strip('"')
pts = pts.rename(columns={"ECTRL ID": "ECTRL_ID"})
labels = labels.rename(columns={"ECTRL ID": "ECTRL_ID"})

pts = pts.merge(labels[["ECTRL_ID", "ADEP", "ADES", "cluster_kmeans"]], on="ECTRL_ID", how="inner")
pts = pts.sort_values(["ECTRL_ID", "Sequence Number"])

print("=== cluster_labels.csv ===")
print(f"rows: {len(labels)}  |  unique ECTRL_IDs: {labels['ECTRL_ID'].nunique()}")
print(labels.groupby(["ADEP", "ADES", "cluster_kmeans"]).size().rename("n_flights").reset_index().to_string(index=False))
print("\n=== after merge ===")
print(f"rows: {len(pts)}  |  unique flights: {pts['ECTRL_ID'].nunique()}")

for (adep, ades), od_pts in pts.groupby(["ADEP", "ADES"]):
    separation = plot_od_representative_routes(od_pts, adep, ades)
    if isinstance(separation, list):
        for pair in separation:
            flag = "  *** BELOW THRESHOLD" if pair["below_threshold"] else ""
            print(f"  C{pair['cluster_a']} vs C{pair['cluster_b']}: {pair['max_deviation_km']:.1f} km{flag}")
