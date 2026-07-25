"""Data-driven flight route alternative and cost characterisation, built for Mercury-style simulators.

The primary deliverable is predict_route_options() - given an aircraft type
and O-D pair, returns ranked route alternatives with predicted cost and
duration, read from the pre-computed full_summary table (see query.py).
"""

from .clustering import cluster_full_dataset, cluster_od_3layer, cluster_summary, merge_similar_clusters
from .costs import EUROCONTROL_RATES, FUEL_KGH, MTOW_TONNES, add_cost_columns
from .data import load_actual_and_filed, load_full_dataset, load_od_counts, load_training_sample
from .features import detect_active_firs, fir_columns, make_route_signatures
from .query import build_full_summary, predict_route_options, query_route_profile
from .validation import cluster_quality, mad_filter, summarise_with_outlier_removal
from .variance import build_actual_metrics, compute_centroid_deltas, compute_self_deltas, haversine_nm

__all__ = [
    "predict_route_options",
    "query_route_profile",
    "build_full_summary",
    "load_training_sample",
    "load_full_dataset",
    "load_od_counts",
    "load_actual_and_filed",
    "detect_active_firs",
    "fir_columns",
    "make_route_signatures",
    "cluster_od_3layer",
    "merge_similar_clusters",
    "cluster_full_dataset",
    "cluster_summary",
    "add_cost_columns",
    "MTOW_TONNES",
    "FUEL_KGH",
    "EUROCONTROL_RATES",
    "build_actual_metrics",
    "compute_centroid_deltas",
    "compute_self_deltas",
    "haversine_nm",
    "cluster_quality",
    "mad_filter",
    "summarise_with_outlier_removal",
]
