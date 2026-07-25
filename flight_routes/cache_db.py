"""SQLite-backed cache for derived pipeline outputs.

Replaces the pile of ad hoc, same-purpose CSV/parquet files that used to
accumulate in cache_dir() (df_sample.csv, df_val.csv, od_counts_full.csv,
df_full.parquet, actual_firs_sample.csv, ...) with one cache.db file, one
table per cached dataset, indexed on whichever columns each function
actually filters or joins on. A real query engine instead of "read the
whole file, then filter in pandas" every time.

SQLite, not a client-server database: this pipeline runs in Colab or on
one laptop, not as a multi-user service, so a client-server DB would be
disproportionate. SQLite is still a genuine relational database (ACID,
SQL, indexes), it just happens to live in a single file with no server
to configure.
"""

import sqlite3

import pandas as pd


def _db_path(cache_dir):
    return cache_dir / "cache.db"


def has_table(cache_dir, table_name):
    path = _db_path(cache_dir)
    if not path.exists():
        return False
    with sqlite3.connect(path) as conn:
        row = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table_name,)
        ).fetchone()
    return row is not None


def read_table(cache_dir, table_name):
    with sqlite3.connect(_db_path(cache_dir)) as conn:
        return pd.read_sql(f'SELECT * FROM "{table_name}"', conn)


def write_table(cache_dir, table_name, df, index_cols=None):
    """Replace table_name with df's contents, then (re)create an index on index_cols.

    method="multi" (batched multi-row INSERTs) is deliberately not used here:
    SQLite caps the number of bound variables per statement (a few hundred to
    a few thousand depending on build), and some cached tables are wide (the
    FIR-distance tables have 300+ columns), so a multi-row insert can exceed
    that cap. The default row-by-row executemany has no such limit.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(_db_path(cache_dir)) as conn:
        df.to_sql(table_name, conn, if_exists="replace", index=False, chunksize=10_000)
        if index_cols:
            cols = ", ".join(f'"{c}"' for c in index_cols)
            idx_name = f"idx_{table_name}_{'_'.join(index_cols)}".replace(" ", "_")
            conn.execute(f'CREATE INDEX IF NOT EXISTS "{idx_name}" ON "{table_name}" ({cols})')
