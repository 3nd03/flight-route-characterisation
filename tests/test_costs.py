import math

import numpy as np
import pandas as pd

from flight_routes.costs import (
    EUROCONTROL_RATES,
    FUEL_KGH,
    JET_A_EUR_PER_KG,
    MTOW_TONNES,
    add_cost_columns,
    detect_ac_col,
    flight_atc_eur,
    flight_fuel_eur,
    _parse_duration,
)


def test_parse_duration_hhmm():
    assert _parse_duration("6:56") == 6 + 56 / 60


def test_parse_duration_numeric_string():
    assert _parse_duration("7.5") == 7.5


def test_flight_atc_eur_single_fir():
    # 1000 nm in LFBBFIR (France, 73.69 EUR/SU), A320-class MTOW 78t
    row = {"mtow_t": 78.0, "LFBBFIR": 1000.0}
    dist_km = 1000.0 * 1.852
    expected = round((dist_km / 100) * math.sqrt(78.0 / 50) * EUROCONTROL_RATES["LFBBFIR"], 2)
    assert flight_atc_eur(row) == expected


def test_flight_atc_eur_missing_mtow_is_nan():
    row = {"mtow_t": float("nan"), "LFBBFIR": 1000.0}
    result = flight_atc_eur(row)
    assert result != result  # NaN != NaN


def test_flight_atc_eur_gander_oceanic_flat_fee_only():
    # CZQXFIR should add the Nav Canada flat fee, not a distance-based charge
    from flight_routes.costs import CAD_EUR, NAV_CANADA_OCEANIC

    row = {"mtow_t": 78.0, "CZQXFIR": 500.0}
    assert flight_atc_eur(row) == round(NAV_CANADA_OCEANIC * CAD_EUR, 2)


def test_flight_fuel_eur():
    row = {"mtow_t": 78.0, "Duration_Hours": "2:00", "ac": "A320"}
    result = flight_fuel_eur(row, ac_col="ac")
    from flight_routes.costs import FUEL_KGH, JET_A_EUR_PER_KG

    assert result == round(FUEL_KGH["A320"] * 2.0 * JET_A_EUR_PER_KG, 2)


def _reference_add_cost_columns(df, ac_col=None):
    """The original row-wise .apply(axis=1) implementation, kept only as a
    test oracle - add_cost_columns itself is now vectorised (it doesn't scale
    to full-dataset size as a row-wise apply, see costs.py docstring)."""
    df = df.copy()
    ac_col = ac_col or detect_ac_col(df)
    df["mtow_t"] = df[ac_col].map(MTOW_TONNES)
    df["atc_eur"] = df.apply(flight_atc_eur, axis=1)
    df["fuel_eur"] = df.apply(lambda row: flight_fuel_eur(row, ac_col), axis=1)
    df["cost_eur"] = (df["atc_eur"] + df["fuel_eur"]).round(2)
    return df


def test_add_cost_columns_vectorised_matches_row_wise_reference():
    rows = [
        {"AC Type": "A320", "Duration_Hours": "2:00", "LFBBFIR": 1000.0, "CZQMFIR": 0.0, "CZQXFIR": 0.0},
        {"AC Type": "B77W", "Duration_Hours": "6:30", "LFBBFIR": 0.0, "CZQMFIR": 400.0, "CZQXFIR": 500.0},
        {"AC Type": "UNKNOWN_TYPE", "Duration_Hours": "1:15", "LFBBFIR": 200.0, "CZQMFIR": 0.0, "CZQXFIR": 0.0},
    ]
    df = pd.DataFrame(rows)

    want = _reference_add_cost_columns(df, ac_col="AC Type")
    got = add_cost_columns(df, ac_col="AC Type")

    for col in ["mtow_t", "atc_eur", "fuel_eur", "cost_eur"]:
        a, b = want[col].to_numpy(dtype=float), got[col].to_numpy(dtype=float)
        assert np.allclose(a, b, atol=0.01, equal_nan=True), col


def test_add_cost_columns_treats_nan_fir_distance_as_zero():
    """Deliberate behaviour difference from the old row-wise version, not a bug:
    the row-wise flight_atc_eur used `row.get(fir, 0) or 0`, which does NOT
    catch NaN (NaN is truthy in Python), so a NaN FIR-distance value silently
    poisoned the whole atc_eur to NaN. This never actually triggered on real
    data (df_sample/df_val have no NaN FIR values - see the exact-match check
    against real data this test file's sibling checks were validated with),
    but it was a latent bug, not intended behaviour: query.build_full_summary's
    _atc_vec already treats a missing/NaN FIR distance as 0 (not crossed),
    and that convention is what produced the published, validated report
    numbers for the full dataset. add_cost_columns now matches that
    convention instead of the row-wise version's dormant bug."""
    df = pd.DataFrame([
        {"AC Type": "A320", "Duration_Hours": "3:00", "LFBBFIR": np.nan, "CZQMFIR": 0.0, "CZQXFIR": 0.0},
    ])
    got = add_cost_columns(df, ac_col="AC Type")
    assert got["atc_eur"].iloc[0] == 0.0
