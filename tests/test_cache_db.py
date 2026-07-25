import pandas as pd

from flight_routes import cache_db


def test_write_then_read_round_trips(tmp_path):
    df = pd.DataFrame({"ECTRL ID": ["1", "2", "3"], "value": [1.5, 2.5, 3.5]})
    cache_db.write_table(tmp_path, "some_table", df, index_cols=["ECTRL ID"])
    out = cache_db.read_table(tmp_path, "some_table")
    assert out[["ECTRL ID", "value"]].equals(df[["ECTRL ID", "value"]])


def test_has_table_false_when_db_missing(tmp_path):
    assert cache_db.has_table(tmp_path, "nonexistent") is False


def test_has_table_false_when_table_missing(tmp_path):
    df = pd.DataFrame({"a": [1]})
    cache_db.write_table(tmp_path, "real_table", df)
    assert cache_db.has_table(tmp_path, "real_table") is True
    assert cache_db.has_table(tmp_path, "other_table") is False


def test_write_table_replaces_existing_contents(tmp_path):
    df1 = pd.DataFrame({"a": [1, 2]})
    df2 = pd.DataFrame({"a": [9, 9, 9]})
    cache_db.write_table(tmp_path, "t", df1)
    cache_db.write_table(tmp_path, "t", df2)
    out = cache_db.read_table(tmp_path, "t")
    assert len(out) == 3
    assert (out["a"] == 9).all()


def test_wide_table_beyond_sqlite_multi_insert_variable_limit(tmp_path):
    # Regression test: to_sql(method="multi") batches many rows into one
    # INSERT with ncols x nrows bound variables, which blows past SQLite's
    # per-statement variable cap on a wide table (this project's FIR-distance
    # tables have 300+ columns). write_table must not hit that.
    wide = pd.DataFrame({f"col{i}": range(50) for i in range(320)})
    cache_db.write_table(tmp_path, "wide", wide, index_cols=["col0"])
    out = cache_db.read_table(tmp_path, "wide")
    assert out.shape == wide.shape
