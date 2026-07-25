import pandas as pd

from flight_routes.features import active_firs_for_group, detect_active_firs, fir_columns, make_route_signatures


def _sample_df():
    return pd.DataFrame({
        "ADEP": ["EGLL", "EGLL", "EGLL", "LEBL"],
        "ADES": ["KJFK", "KJFK", "KJFK", "LEPA"],
        "EGTTFIR": [100, 100, 0, 0],
        "EGGXFIR": [500, 500, 500, 0],
        "LECBFIR": [0, 0, 0, 50],
        "Latitude": [1, 2, 3, 4],  # not a FIR column, must be excluded
    })


def test_fir_columns_excludes_non_fir_columns():
    cols = fir_columns(_sample_df())
    assert "Latitude" not in cols
    assert set(cols) == {"EGTTFIR", "EGGXFIR", "LECBFIR"}


def test_active_firs_for_group_threshold():
    df = _sample_df()
    egll_kjfk = df[df["ADEP"] == "EGLL"]
    # EGTTFIR crossed by 2/3 flights (>=0.25), EGGXFIR by 3/3, LECBFIR by 0/3
    active = active_firs_for_group(egll_kjfk, fir_columns(df))
    assert set(active) == {"EGTTFIR", "EGGXFIR"}


def test_detect_active_firs_is_union_across_pairs():
    df = _sample_df()
    active = detect_active_firs(df, fir_columns(df))
    assert set(active) == {"EGTTFIR", "EGGXFIR", "LECBFIR"}


def test_make_route_signatures_is_binary():
    df = _sample_df()
    sigs = make_route_signatures(df, fir_columns(df))
    assert set(sigs.values.ravel()) <= {0, 1}
    assert sigs.loc[0, "EGTTFIR"] == 1
    assert sigs.loc[2, "EGTTFIR"] == 0
