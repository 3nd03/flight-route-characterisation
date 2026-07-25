import numpy as np
import pandas as pd

from flight_routes.validation import mad_filter, summarise_with_outlier_removal


def test_mad_filter_removes_obvious_outlier():
    clean, n_removed = mad_filter([10, 11, 9, 10, 500, 10, 9])
    assert n_removed == 1
    assert 500 not in clean


def test_mad_filter_empty_input():
    clean, n_removed = mad_filter([])
    assert len(clean) == 0
    assert n_removed == 0


def test_mad_filter_zero_mad_keeps_everything():
    # all identical values -> MAD is 0 -> nothing flagged
    clean, n_removed = mad_filter([5, 5, 5, 5])
    assert n_removed == 0
    assert len(clean) == 4


def test_summarise_with_outlier_removal_strip_suffix():
    df = pd.DataFrame({
        "ADEP": ["EGLL"] * 5,
        "ADES": ["KJFK"] * 5,
        "cluster_kmeans": [0] * 5,
        "delta_dist_nm_pooled": [1.0, 2.0, 1.5, 2.5, 1.0],
    })
    out = summarise_with_outlier_removal(
        df, ["ADEP", "ADES", "cluster_kmeans"], ["delta_dist_nm_pooled"], strip_suffix="_pooled"
    )
    assert "delta_dist_nm_mean" in out.columns
    assert "delta_dist_nm_pooled_mean" not in out.columns
    assert out.loc[0, "n"] == 5
