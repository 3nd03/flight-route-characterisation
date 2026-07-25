"""En-route ATC charge and fuel cost model.

Rates verified against EUROCONTROL's September 2023 monthly adjusted unit
rate table (see docs/report_draft.md Discussion for the audit that found
and corrected 10 of the original 18 entries).
"""

import numpy as np
import pandas as pd

MTOW_TONNES = {
    "B77W": 347.4, "B772": 297.6, "B773": 299.4, "B77L": 347.4,
    "A35K": 280.0, "A359": 280.0, "A21N": 97.0, "A321": 93.5,
    "A320": 78.0, "A20N": 79.0, "A319": 75.5, "A19N": 75.5,
    "B738": 79.016, "B737": 65.3, "B739": 85.1, "BCS3": 70.9,
    "AT43": 16.9, "B38M": 82.2, "A339": 251.0, "A333": 242.0,
    "CRJ9": 38.3, "E195": 52.3, "B764": 204.1, "DH8B": 16.5,
    "B789": 254.0,
    "AT76": 23.0, "AT75": 22.8, "AT45": 18.6, "DH8D": 29.6, "DH8A": 15.7,
    "E190": 51.8, "E170": 37.2, "E75L": 40.4, "E75S": 38.8,
    "E295": 61.5, "E145": 22.0, "CRJX": 38.3,
    "A332": 242.0, "A388": 575.0, "B788": 227.9, "B78X": 254.0,
    "B763": 186.9, "B752": 115.7, "B734": 68.0, "SU95": 49.5,
}

# Cruise fuel flow (kg/h) - ICAO FEAT / EcoScope block-hour averages at typical cruise
FUEL_KGH = {
    "B77W": 7800, "B772": 6600, "B773": 7000, "B77L": 7800,
    "A35K": 5800, "A359": 5800, "A339": 5400, "A333": 6200,
    "B764": 5500, "B789": 5500,
    "A21N": 2500, "A321": 2800, "A320": 2500, "A20N": 2300,
    "A319": 2200, "A19N": 2000,
    "B38M": 2100, "B738": 2400, "B737": 2200, "B739": 2600,
    "BCS3": 1900, "CRJ9": 1400, "E195": 1800,
    "AT43": 600, "DH8B": 500,
    "AT76": 900, "AT75": 850, "AT45": 650,
    "DH8D": 1150, "DH8A": 600,
    "E190": 2200, "E170": 1700, "E75L": 1800, "E75S": 1800,
    "E295": 2600, "E145": 1350, "CRJX": 2000,
    "A332": 6000, "A388": 12000,
    "B788": 5200, "B78X": 6500, "B763": 5500, "B752": 4000,
    "B734": 3100, "SU95": 2300,
}

JET_A_EUR_PER_KG = 0.81  # EIA US Gulf Coast FOB avg Sep 2023, EUR/USD 1.074

# EUROCONTROL/CRCO en-route unit rates, EUR per service unit (Sep 2023, verified)
EUROCONTROL_RATES = {
    "BGGLFIR": 61.04, "EGGXFIR": 87.88, "EGTTFIR": 87.88, "EGTTUIR": 87.88,
    "EISNUIR": 26.46, "ENORFIR": 47.66,
    "LECBFIR": 54.71, "LPPOFIR": 10.03, "LPPCFIR": 47.39,
    "LFBBFIR": 73.69, "LFFFFIR": 73.69, "LFEEFIR": 73.69,
    "EDGGFIR": 73.04, "EDMMFIR": 73.04, "EDWWFIR": 73.04,
    "LGGGFIR": 25.54, "LGGGUIR": 25.54, "LIRRUIR": 72.37, "LIMMUIR": 72.37,
    "LFFFUIR": 73.69, "LJLAFIR": 65.32, "LOVVFIR": 66.91, "LDZOFIR": 45.83,
    "LSASUIR": 120.92, "EBURUIR": 113.21, "EDUUUIR": 73.04,
    "LAAAFIR": 55.71, "LYBAUIR": 39.50,
    "EGPXUIR": 87.88, "BIRDFIR": 66.80,
}

# Nav Canada 2024 rates used as a proxy for 2023 (<3% annual change)
NAV_CANADA_R       = 0.03402  # CAD / (km x sqrt(tonne)) - domestic enroute formula
NAV_CANADA_OCEANIC = 210.19   # CAD flat fee - Gander Oceanic
CAD_EUR            = 0.695    # Aug 2023 average exchange rate

# CZQXFIR (Gander Oceanic): flat fee only, excluded from the distance formula.
# CZQMFIR (Moncton) / CZULFIR (Montreal): distance-based Nav Canada formula.
# KZBWFIR/UIR, KZNYFIR, KZWYFIR: zero - no FAA overflight fee for US-landing flights.


def _parse_duration(s):
    """Parse an 'H:MM' duration string (or a bare number of hours) into float hours."""
    if isinstance(s, str) and ":" in s:
        h, m = s.split(":")
        return int(h) + int(m) / 60
    return float(s)


def detect_ac_col(df):
    """Find the aircraft-type column, whatever it's actually named in this dataset."""
    return next(c for c in df.columns if "AC Type" in c)


def rate_firs_in(df):
    """EUROCONTROL_RATES keys plus the three Nav Canada FIRs, restricted to columns df actually has."""
    return [f for f in list(EUROCONTROL_RATES.keys()) + ["CZQMFIR", "CZULFIR", "CZQXFIR"]
            if f in df.columns]


def flight_atc_eur(row, mtow_col="mtow_t"):
    """EUROCONTROL + Nav Canada en-route ATC charge for one flight (a DataFrame row)."""
    mtow = row[mtow_col]
    if pd.isna(mtow):
        return np.nan

    # Sum FIR + UIR distances per ANSP base code: same charging zone, different altitude bands
    base_dists, base_rates = {}, {}
    for fir, rate in EUROCONTROL_RATES.items():
        d = row.get(fir, 0) or 0
        base = fir[:-3]
        base_dists[base] = base_dists.get(base, 0) + d
        base_rates.setdefault(base, rate)

    total = 0.0
    for base, d_nm in base_dists.items():
        if d_nm <= 0:
            continue
        dist_km = d_nm * 1.852
        su = (dist_km / 100) * np.sqrt(mtow / 50)
        total += su * base_rates[base]

    for fir in ("CZQMFIR", "CZULFIR"):
        dist_nm = row.get(fir, 0)
        if dist_nm > 0:
            dist_km = dist_nm * 1.852
            total += NAV_CANADA_R * np.sqrt(mtow) * dist_km * CAD_EUR
    if row.get("CZQXFIR", 0) > 0:
        total += NAV_CANADA_OCEANIC * CAD_EUR
    return round(total, 2)


def flight_fuel_eur(row, ac_col, duration_col="Duration_Hours"):
    """Fuel cost for one flight: cruise fuel flow x duration x jet fuel price."""
    fuel_kgh = FUEL_KGH.get(row[ac_col])
    if fuel_kgh is None or pd.isna(row["mtow_t"]):
        return np.nan
    return round(fuel_kgh * _parse_duration(row[duration_col]) * JET_A_EUR_PER_KG, 2)


def add_cost_columns(df, ac_col=None):
    """Add mtow_t, atc_eur, fuel_eur, cost_eur columns. Returns a new DataFrame.

    Vectorised across all rows at once, not a row-wise `.apply(axis=1)` (the
    row-wise version this replaced is still available as flight_atc_eur/
    flight_fuel_eur for single-row use, e.g. inside variance.build_actual_metrics
    where it's cheap relative to the point-level work around it). At
    full-dataset scale (536,520 rows, 300+ FIR columns) `.apply(axis=1)` risked
    exhausting Colab's RAM - this mirrors the vectorised approach already used
    (and validated against the published report numbers) in
    query.build_full_summary. Validated to match the row-wise version exactly
    on the real training sample, see tests/test_costs.py.
    """
    df = df.copy()
    ac_col = ac_col or detect_ac_col(df)
    df["mtow_t"] = df[ac_col].map(MTOW_TONNES)
    mtow = df["mtow_t"]

    total = pd.Series(0.0, index=df.index)
    base_totals, base_rate_map = {}, {}
    for fir, rate in EUROCONTROL_RATES.items():
        if fir not in df.columns:
            continue
        base = fir[:-3]
        base_totals[base] = base_totals.get(base, pd.Series(0.0, index=df.index)) + df[fir].fillna(0)
        base_rate_map.setdefault(base, rate)
    for base, dist_nm_col in base_totals.items():
        dist_km = dist_nm_col * 1.852
        total += (dist_km / 100) * np.sqrt(mtow / 50) * base_rate_map[base]
    for fir in ("CZQMFIR", "CZULFIR"):
        if fir in df.columns:
            dist_km = df[fir].fillna(0) * 1.852
            total += NAV_CANADA_R * np.sqrt(mtow) * dist_km * CAD_EUR
    if "CZQXFIR" in df.columns:
        total += (df["CZQXFIR"].fillna(0) > 0).astype(float) * NAV_CANADA_OCEANIC * CAD_EUR
    df["atc_eur"] = total.where(mtow.notna()).round(2)

    duration_h = df["Duration_Hours"].apply(_parse_duration)
    fuel_kgh = df[ac_col].map(FUEL_KGH)
    df["fuel_eur"] = (fuel_kgh * duration_h * JET_A_EUR_PER_KG).where(fuel_kgh.notna() & mtow.notna()).round(2)

    df["cost_eur"] = (df["atc_eur"] + df["fuel_eur"]).round(2)
    return df
