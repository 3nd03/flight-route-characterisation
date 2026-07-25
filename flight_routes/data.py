"""Loading, joining and caching flight-plan data.

Paths are configurable via environment variables so the same code works
against a local ``data/`` folder, a mounted Google Drive path in Colab, or
someone else's raw-data archive:

- ``FLIGHT_ROUTES_RAW_DIR``   - raw EUROCONTROL parquet/csv exports (large, not in git)
- ``FLIGHT_ROUTES_CACHE_DIR`` - derived/cached outputs (small, safe to commit)

Both default to ``<package root>/data/{raw,processed}``. Read fresh on every
call (not frozen at import time) - set the env var any time before calling
a function, including after ``import flight_routes`` has already run, e.g.
after mounting Drive in a later notebook cell.
"""

import os
from pathlib import Path

import pandas as pd

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def raw_dir():
    return Path(os.environ.get("FLIGHT_ROUTES_RAW_DIR", _PACKAGE_ROOT / "data" / "raw"))


def cache_dir():
    return Path(os.environ.get("FLIGHT_ROUTES_CACHE_DIR", _PACKAGE_ROOT / "data" / "processed"))


TARGET_OD = [("LEBL", "LEPA"), ("EGLL", "KJFK"), ("LPPT", "EDDB")]
VAL_OD    = ("EGLL", "LGAV")


def _inner_join_scheduled(fir_df, flights_df):
    """Inner-join the FIR-distance and flights datasets on ECTRL ID, scheduled flights only."""
    flights_df = flights_df[flights_df["ICAO Flight Type"] == "S"]
    flights_df = flights_df.drop(columns=["STATFOR Market Segment"], errors="ignore")
    df = fir_df.merge(flights_df, on="ECTRL ID", how="inner", suffixes=("", "_drop"))
    df = df.drop(columns=[c for c in df.columns if c.endswith("_drop")])
    return df


def load_training_sample(target_od=TARGET_OD, val_od=VAL_OD, force_rebuild=False):
    """Return (df_sample, df_val) for the training O-D pairs and the held-out validation pair.

    Loads from cache_dir() if present; otherwise rebuilds from raw_dir() parquets
    and writes the cache for next time.
    """
    cache = cache_dir()
    sample_cache = cache / "df_sample.csv"
    val_cache    = cache / "df_val.csv"

    if sample_cache.exists() and val_cache.exists() and not force_rebuild:
        return pd.read_csv(sample_cache), pd.read_csv(val_cache)

    raw = raw_dir()
    fir     = pd.read_parquet(raw / "Final_Wide_Report.parquet")
    flights = pd.read_parquet(raw / "Flights_20230901_20230930.parquet")
    df = _inner_join_scheduled(fir, flights)
    del fir, flights

    od_idx    = pd.MultiIndex.from_frame(df[["ADEP", "ADES"]])
    df_sample = df[od_idx.isin(target_od)].reset_index(drop=True)
    df_val    = df[od_idx.isin([val_od])].reset_index(drop=True)
    del df

    cache.mkdir(parents=True, exist_ok=True)
    df_sample.to_csv(sample_cache, index=False)
    df_val.to_csv(val_cache, index=False)
    return df_sample, df_val


def load_od_counts(force_rebuild=False):
    """Flight counts per O-D pair across the full scheduled-flights dataset."""
    cache = cache_dir()
    cache_file = cache / "od_counts_full.csv"
    if cache_file.exists() and not force_rebuild:
        return pd.read_csv(cache_file)

    flights = pd.read_parquet(
        raw_dir() / "Flights_20230901_20230930.parquet",
        columns=["ECTRL ID", "ADEP", "ADES", "ICAO Flight Type"],
    )
    flights = flights[flights["ICAO Flight Type"] == "S"]
    od_counts = (
        flights.groupby(["ADEP", "ADES"])
        .size()
        .reset_index(name="n_flights")
        .sort_values("n_flights", ascending=False)
        .reset_index(drop=True)
    )
    cache.mkdir(parents=True, exist_ok=True)
    od_counts.to_csv(cache_file, index=False)
    return od_counts


def load_full_dataset(min_flights_per_od=30, force_rebuild=False):
    """Full FIR+Flights join, filtered to O-D pairs with at least ``min_flights_per_od`` flights."""
    cache = cache_dir()
    cache_file = cache / "df_full.parquet"
    if cache_file.exists() and not force_rebuild:
        return pd.read_parquet(cache_file)

    raw = raw_dir()
    fir     = pd.read_parquet(raw / "Final_Wide_Report.parquet")
    flights = pd.read_parquet(raw / "Flights_20230901_20230930.parquet")
    df = _inner_join_scheduled(fir, flights)
    del fir, flights

    pair_counts = df.groupby(["ADEP", "ADES"]).size()
    qualifying  = pair_counts[pair_counts >= min_flights_per_od].index
    od_idx      = pd.MultiIndex.from_frame(df[["ADEP", "ADES"]])
    df_full     = df[od_idx.isin(qualifying)].reset_index(drop=True)
    del df

    cache.mkdir(parents=True, exist_ok=True)
    df_full.to_parquet(cache_file, index=False)
    return df_full


def _load_filtered_csv(raw_path, sample_ids, chunksize=500_000):
    """Read a large CSV in chunks, keeping only rows for the given ECTRL IDs."""
    chunks = [
        chunk[chunk["ECTRL ID"].isin(sample_ids)]
        for chunk in pd.read_csv(raw_path, dtype={"ECTRL ID": str}, chunksize=chunksize)
    ]
    return pd.concat(chunks, ignore_index=True)


def load_actual_and_filed(df_sample, force_rebuild=False):
    """Return (actual_firs, actual_pts, filed_pts), filtered to the flights in df_sample.

    Reads from raw_dir(): Flight_FIRs_Actual, Flight_Points_Actual, Flight_Points_Filed
    (each ~500MB-2GB, per the September 2023 EUROCONTROL export).
    """
    cache = cache_dir()
    raw = raw_dir()
    actual_firs_cache = cache / "actual_firs_sample.csv"
    actual_pts_cache  = cache / "actual_points_sample.csv"
    filed_pts_cache   = cache / "filed_points_sample.csv"
    sample_ids = set(df_sample["ECTRL ID"].astype(str))

    if actual_firs_cache.exists() and actual_pts_cache.exists() and not force_rebuild:
        actual_firs = pd.read_csv(actual_firs_cache, dtype={"ECTRL ID": str})
        actual_pts  = pd.read_csv(actual_pts_cache, dtype={"ECTRL ID": str})
    else:
        actual_firs_raw = pd.read_csv(
            raw / "Flight_FIRs_Actual_20230901_20230930.csv", dtype={"ECTRL ID": str}
        )
        actual_firs = actual_firs_raw[actual_firs_raw["ECTRL ID"].isin(sample_ids)].reset_index(drop=True)
        del actual_firs_raw
        actual_pts = _load_filtered_csv(raw / "Flight_Points_Actual_20230901_20230930.csv", sample_ids)

        cache.mkdir(parents=True, exist_ok=True)
        actual_firs.to_csv(actual_firs_cache, index=False)
        actual_pts.to_csv(actual_pts_cache, index=False)

    if filed_pts_cache.exists() and not force_rebuild:
        filed_pts = pd.read_csv(filed_pts_cache, dtype={"ECTRL ID": str})
    else:
        filed_pts = _load_filtered_csv(raw / "Flight_Points_Filed_20230901_20230930.csv", sample_ids)
        cache.mkdir(parents=True, exist_ok=True)
        filed_pts.to_csv(filed_pts_cache, index=False)

    return actual_firs, actual_pts, filed_pts
