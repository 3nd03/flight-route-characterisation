"""Validates build_actual_metrics_full_dataset (the memory-bounded, batched
path used at full-dataset scale) against calling build_actual_metrics
directly on the same data in one shot - batching must not change results,
drop flights, or duplicate them across batch boundaries.
"""

import numpy as np
import pandas as pd
import pytest

from flight_routes import data
from flight_routes.costs import MTOW_TONNES
from flight_routes.variance import build_actual_metrics, build_actual_metrics_full_dataset

EUROCONTROL_RATES = {"FIRA": 50.0, "FIRB": 60.0}
MONTH = "TESTMONTH"
N_FLIGHTS = 47  # deliberately not a multiple of batch_size, to exercise a partial last batch


def _write_synthetic_raw_files(raw_dir):
    firs_rows, pts_rows, filed_rows, sample_rows = [], [], [], []
    for eid in range(1, N_FLIGHTS + 1):
        fir = "FIRA" if eid % 2 == 0 else "FIRB"
        firs_rows.append({"ECTRL ID": eid, "FIR ID": fir,
                           "Entry Time": "01-09-2023 10:00:00", "Exit Time": "01-09-2023 12:00:00"})
        base_lat, base_lon = 50.0 + eid * 0.01, eid * 0.01
        for i, minute in enumerate([0, 30, 60]):
            ts = f"01-09-2023 {10 + minute // 60:02d}:{minute % 60:02d}:00"
            pts_rows.append({"ECTRL ID": eid, "Time Over": ts,
                              "Latitude": base_lat + i * 0.1, "Longitude": base_lon + i * 0.1})
            filed_rows.append({"ECTRL ID": eid, "Sequence Number": i,
                                "Latitude": base_lat + i * 0.1, "Longitude": base_lon + i * 0.1})
        sample_rows.append({"ECTRL ID": eid, "AC Type": "A320"})

    raw_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(firs_rows).to_csv(raw_dir / f"Flight_FIRs_Actual_{MONTH}.csv", index=False)
    pd.DataFrame(pts_rows).to_csv(raw_dir / f"Flight_Points_Actual_{MONTH}.csv", index=False)
    pd.DataFrame(filed_rows).to_csv(raw_dir / f"Flight_Points_Filed_{MONTH}.csv", index=False)
    return pd.DataFrame(sample_rows)


@pytest.fixture
def synthetic_env(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    cache_dir = tmp_path / "processed"
    df_full = _write_synthetic_raw_files(raw_dir)
    monkeypatch.setenv("FLIGHT_ROUTES_RAW_DIR", str(raw_dir))
    monkeypatch.setenv("FLIGHT_ROUTES_CACHE_DIR", str(cache_dir))
    return df_full


def test_batched_matches_unbatched_reference(synthetic_env):
    df_full = synthetic_env

    batched = build_actual_metrics_full_dataset(
        df_full, EUROCONTROL_RATES, ac_col="AC Type", batch_size=10,
        force_rebuild=True, month=MONTH, progress_every=2,
    )

    raw = data.raw_dir()
    actual_firs = pd.read_csv(raw / f"Flight_FIRs_Actual_{MONTH}.csv", dtype={"ECTRL ID": str})
    actual_pts  = pd.read_csv(raw / f"Flight_Points_Actual_{MONTH}.csv", dtype={"ECTRL ID": str})
    filed_pts   = pd.read_csv(raw / f"Flight_Points_Filed_{MONTH}.csv", dtype={"ECTRL ID": str})
    df_full_ref = df_full.copy()
    df_full_ref["ECTRL ID"] = df_full_ref["ECTRL ID"].astype(str)
    df_full_ref["mtow_t"] = df_full_ref["AC Type"].map(MTOW_TONNES)
    unbatched = build_actual_metrics(df_full_ref, actual_firs, actual_pts, filed_pts,
                                      EUROCONTROL_RATES, ac_col="AC Type")

    assert set(batched["ECTRL ID"]) == set(unbatched["ECTRL ID"]) == set(range(1, N_FLIGHTS + 1))
    assert len(batched) == N_FLIGHTS  # no duplicates across batch boundaries

    batched = batched.sort_values("ECTRL ID").reset_index(drop=True)
    unbatched = unbatched.sort_values("ECTRL ID").reset_index(drop=True)
    for col in ["actual_duration_h", "actual_total_dist_nm", "actual_atc_eur",
                "actual_fuel_eur", "actual_fuel_kg", "actual_cost_eur", "planned_dist_nm"]:
        assert np.allclose(batched[col].to_numpy(dtype=float), unbatched[col].to_numpy(dtype=float),
                            atol=0.05, equal_nan=True), col


def test_batched_result_is_cached(synthetic_env, monkeypatch):
    df_full = synthetic_env
    build_actual_metrics_full_dataset(df_full, EUROCONTROL_RATES, ac_col="AC Type",
                                       batch_size=10, force_rebuild=True, month=MONTH)
    cache = data.cache_dir()
    from flight_routes import cache_db
    assert cache_db.has_table(cache, "actual_metrics_full")

    # second call with force_rebuild=False should read the cache, not touch raw_dir at all
    monkeypatch.setenv("FLIGHT_ROUTES_RAW_DIR", str(data.raw_dir()) + "_does_not_exist")
    out = build_actual_metrics_full_dataset(df_full, EUROCONTROL_RATES, ac_col="AC Type",
                                             batch_size=10, force_rebuild=False, month=MONTH)
    assert len(out) == N_FLIGHTS


def test_batch_tmp_dir_cleaned_up(synthetic_env):
    df_full = synthetic_env
    build_actual_metrics_full_dataset(df_full, EUROCONTROL_RATES, ac_col="AC Type",
                                       batch_size=10, force_rebuild=True, month=MONTH)
    assert not (data.cache_dir() / "_variance_batches").exists()
