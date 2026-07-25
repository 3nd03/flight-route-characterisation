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

Cached outputs live in one SQLite file (cache_dir()/cache.db, see
cache_db.py), not scattered CSV/parquet files.

Every function below takes a ``month`` string (default DEFAULT_MONTH,
the Sep 2023 export this project was built and validated against) that's
substituted into the raw EUROCONTROL filenames, so pointing FLIGHT_ROUTES_RAW_DIR
and month at a different month's export should work unchanged - this has
not been run against another month, since no other month's raw data has
been available to test with. If repeating the pipeline for another month,
also point FLIGHT_ROUTES_CACHE_DIR at a fresh folder: the cache is keyed by
cache_dir() alone, not by month, so reusing the same cache dir for a
second month overwrites the first month's cached tables.
"""

import os
from pathlib import Path

import pandas as pd

from . import cache_db

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MONTH = "20230901_20230930"


def raw_dir():
    return Path(os.environ.get("FLIGHT_ROUTES_RAW_DIR", _PACKAGE_ROOT / "data" / "raw"))


def cache_dir():
    return Path(os.environ.get("FLIGHT_ROUTES_CACHE_DIR", _PACKAGE_ROOT / "data" / "processed"))


TARGET_OD = [("LEBL", "LEPA"), ("EGLL", "KJFK"), ("LPPT", "EDDB")]
VAL_OD    = ("EGLL", "LGAV")


def _flights_path(raw, month):
    return raw / f"Flights_{month}.parquet"


def _fir_report_path(raw, fir_report_name="Final_Wide_Report.parquet"):
    # Unlike the other raw exports, the FIR-distance wide report's filename
    # doesn't embed the month in this dataset's naming convention.
    return raw / fir_report_name


def _firs_actual_path(raw, month):
    return raw / f"Flight_FIRs_Actual_{month}.csv"


def _points_actual_path(raw, month):
    return raw / f"Flight_Points_Actual_{month}.csv"


def _points_filed_path(raw, month):
    return raw / f"Flight_Points_Filed_{month}.csv"


def _inner_join_scheduled(fir_df, flights_df):
    """Inner-join the FIR-distance and flights datasets on ECTRL ID, scheduled flights only."""
    flights_df = flights_df[flights_df["ICAO Flight Type"] == "S"]
    flights_df = flights_df.drop(columns=["STATFOR Market Segment"], errors="ignore")
    df = fir_df.merge(flights_df, on="ECTRL ID", how="inner", suffixes=("", "_drop"))
    df = df.drop(columns=[c for c in df.columns if c.endswith("_drop")])
    return df


def load_training_sample(target_od=TARGET_OD, val_od=VAL_OD, force_rebuild=False, month=DEFAULT_MONTH):
    """Return (df_sample, df_val) for the training O-D pairs and the held-out validation pair.

    Loads from cache_dir()'s cache.db if present; otherwise rebuilds from
    raw_dir() parquets and writes the cache for next time.
    """
    cache = cache_dir()
    if not force_rebuild and cache_db.has_table(cache, "df_sample") and cache_db.has_table(cache, "df_val"):
        return cache_db.read_table(cache, "df_sample"), cache_db.read_table(cache, "df_val")

    raw = raw_dir()
    fir     = pd.read_parquet(_fir_report_path(raw))
    flights = pd.read_parquet(_flights_path(raw, month))
    df = _inner_join_scheduled(fir, flights)
    del fir, flights

    od_idx    = pd.MultiIndex.from_frame(df[["ADEP", "ADES"]])
    df_sample = df[od_idx.isin(target_od)].reset_index(drop=True)
    df_val    = df[od_idx.isin([val_od])].reset_index(drop=True)
    del df

    cache_db.write_table(cache, "df_sample", df_sample, index_cols=["ECTRL ID"])
    cache_db.write_table(cache, "df_val", df_val, index_cols=["ECTRL ID"])
    return df_sample, df_val


def load_od_counts(force_rebuild=False, month=DEFAULT_MONTH):
    """Flight counts per O-D pair across the full scheduled-flights dataset."""
    cache = cache_dir()
    if not force_rebuild and cache_db.has_table(cache, "od_counts_full"):
        return cache_db.read_table(cache, "od_counts_full")

    flights = pd.read_parquet(
        _flights_path(raw_dir(), month),
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
    cache_db.write_table(cache, "od_counts_full", od_counts, index_cols=["ADEP", "ADES"])
    return od_counts


def load_full_dataset(min_flights_per_od=30, force_rebuild=False, month=DEFAULT_MONTH):
    """Full FIR+Flights join, filtered to O-D pairs with at least ``min_flights_per_od`` flights.

    Cached as parquet, not via cache_db/SQLite like the other loaders here:
    df_full is a wide numeric matrix (300+ FIR-distance columns) at full
    scale (536,520 rows), and it's always consumed whole (cluster_full_dataset
    groups the entire frame, nothing ever queries it by a WHERE-clause-style
    filter), so SQLite's indexing has nothing to offer here while its
    row-oriented read path costs a lot: benchmarked at 238s to read this
    table back from SQLite vs 1.7s from parquet on the real Sep 2023 dataset,
    a ~140x regression for zero benefit. Parquet's columnar layout is the
    right tool for this one specifically.
    """
    cache = cache_dir()
    cache_file = cache / "df_full.parquet"
    if cache_file.exists() and not force_rebuild:
        return pd.read_parquet(cache_file)

    raw = raw_dir()
    fir     = pd.read_parquet(_fir_report_path(raw))
    flights = pd.read_parquet(_flights_path(raw, month))
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


def _partition_by_batch(raw_path, id_to_batch, tmp_dir, prefix, chunksize=500_000):
    """Single chunked pass over raw_path, appending each matching row to its batch's own CSV.

    Splits a multi-GB raw file into batch-sized pieces on disk without ever
    holding more than one chunk in RAM, so a later per-batch loop (see
    variance.build_actual_metrics_full_dataset) never needs the whole file
    in memory at once, only one batch at a time.
    """
    batch_paths = {}
    for chunk in pd.read_csv(raw_path, dtype={"ECTRL ID": str}, chunksize=chunksize):
        chunk = chunk[chunk["ECTRL ID"].isin(id_to_batch)]
        if chunk.empty:
            continue
        batch_num = chunk["ECTRL ID"].map(id_to_batch)
        for b, sub in chunk.groupby(batch_num):
            path = tmp_dir / f"{prefix}_batch{b}.csv"
            sub.to_csv(path, mode="a", index=False, header=not path.exists())
            batch_paths.setdefault(b, path)
    return batch_paths


def load_actual_and_filed(df_sample, force_rebuild=False, month=DEFAULT_MONTH):
    """Return (actual_firs, actual_pts, filed_pts), filtered to the flights in df_sample.

    Reads from raw_dir(): Flight_FIRs_Actual, Flight_Points_Actual, Flight_Points_Filed
    (each ~500MB-2GB, per the September 2023 EUROCONTROL export). Materialises
    every matching row of all three files in RAM at once - fine at training-sample
    scale (~1,650 flights) but not for the full dataset (536,520 flights, ~60% of
    each raw file); see variance.build_actual_metrics_full_dataset for the
    memory-bounded, batched equivalent used at that scale.
    """
    cache = cache_dir()
    raw = raw_dir()
    sample_ids = set(df_sample["ECTRL ID"].astype(str))

    have_firs_pts = (
        not force_rebuild
        and cache_db.has_table(cache, "actual_firs_sample")
        and cache_db.has_table(cache, "actual_points_sample")
    )
    if have_firs_pts:
        actual_firs = cache_db.read_table(cache, "actual_firs_sample")
        actual_pts  = cache_db.read_table(cache, "actual_points_sample")
    else:
        actual_firs_raw = pd.read_csv(_firs_actual_path(raw, month), dtype={"ECTRL ID": str})
        actual_firs = actual_firs_raw[actual_firs_raw["ECTRL ID"].isin(sample_ids)].reset_index(drop=True)
        del actual_firs_raw
        actual_pts = _load_filtered_csv(_points_actual_path(raw, month), sample_ids)

        cache_db.write_table(cache, "actual_firs_sample", actual_firs, index_cols=["ECTRL ID"])
        cache_db.write_table(cache, "actual_points_sample", actual_pts, index_cols=["ECTRL ID"])

    if not force_rebuild and cache_db.has_table(cache, "filed_points_sample"):
        filed_pts = cache_db.read_table(cache, "filed_points_sample")
    else:
        filed_pts = _load_filtered_csv(_points_filed_path(raw, month), sample_ids)
        cache_db.write_table(cache, "filed_points_sample", filed_pts, index_cols=["ECTRL ID"])

    return actual_firs, actual_pts, filed_pts
