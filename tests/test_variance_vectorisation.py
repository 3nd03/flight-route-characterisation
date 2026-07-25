"""Validates the vectorised build_actual_metrics against the original
per-flight-loop implementation it replaced (kept here only as a reference
oracle, not part of the shipped library), across the edge cases that make
the FIR-attribution logic tricky: a segment entirely inside one FIR, a
segment that crosses from one FIR into another (time-proportional split),
a segment that starts in a gap with no FIR match at all (distance dropped,
not assigned to anything), and UIR/FIR consolidation for the same ANSP.
"""

import numpy as np
import pandas as pd
import pytest

from flight_routes.variance import build_actual_metrics, haversine_nm

EUROCONTROL_RATES = {"FIRA": 50.0, "FIRB": 60.0, "ABCDFIR": 70.0}


# --- reference implementation (the original per-flight loop, pre-vectorisation) ---

def _ref_flight_actual_metrics(firs_grp, pts_grp, eurocontrol_rates):
    f = firs_grp.copy()
    f["entry_dt"] = pd.to_datetime(f["Entry Time"], dayfirst=True)
    f["exit_dt"] = pd.to_datetime(f["Exit Time"], dayfirst=True)
    airborne = (f[~f["FIR ID"].isin(["TAXI_OUT", "TAXI_IN"])]
                .sort_values("entry_dt").reset_index(drop=True))
    if airborne.empty:
        return {}, np.nan, 0.0

    dur_h = (airborne["exit_dt"].iloc[-1] - airborne["entry_dt"].iloc[0]).total_seconds() / 3600

    p = pts_grp.copy()
    p["time_dt"] = pd.to_datetime(p["Time Over"], dayfirst=True)
    p = (p[(p["time_dt"] >= airborne["entry_dt"].iloc[0]) &
           (p["time_dt"] <= airborne["exit_dt"].iloc[-1])]
         .sort_values("time_dt").reset_index(drop=True))
    if len(p) < 2:
        return {}, dur_h, 0.0

    entries = airborne["entry_dt"].values.astype("datetime64[ns]")
    exits = airborne["exit_dt"].values.astype("datetime64[ns]")
    fids = airborne["FIR ID"].values
    lats = p["Latitude"].values.astype(float)
    lons = p["Longitude"].values.astype(float)
    times = p["time_dt"].values.astype("datetime64[ns]")

    def _idx(t):
        for i in range(len(entries)):
            if entries[i] <= t <= exits[i]:
                return i
        return -1

    fir_dists, total_dist = {}, 0.0
    for i in range(len(p) - 1):
        d = float(haversine_nm(lats[i], lons[i], lats[i + 1], lons[i + 1]))
        total_dist += d
        i1, i2 = _idx(times[i]), _idx(times[i + 1])
        if i1 == i2:
            if i1 >= 0:
                fir_dists[fids[i1]] = fir_dists.get(fids[i1], 0) + d
        elif i1 >= 0:
            dt = float((times[i + 1] - times[i]) / np.timedelta64(1, "s"))
            if dt > 0:
                frac = max(0.0, min(1.0, float((exits[i1] - times[i]) / np.timedelta64(1, "s")) / dt))
                fir_dists[fids[i1]] = fir_dists.get(fids[i1], 0) + d * frac
                if i2 >= 0:
                    fir_dists[fids[i2]] = fir_dists.get(fids[i2], 0) + d * (1 - frac)
    for fid in list(fir_dists.keys()):
        if fid.endswith("UIR"):
            base_fir = fid[:-3] + "FIR"
            if base_fir in eurocontrol_rates:
                fir_dists[base_fir] = fir_dists.get(base_fir, 0) + fir_dists.pop(fid)
    return fir_dists, dur_h, total_dist


def _ref_total_dist_from_pts(pts_grp):
    p = pts_grp.sort_values("Sequence Number").reset_index(drop=True)
    lats = p["Latitude"].values.astype(float)
    lons = p["Longitude"].values.astype(float)
    total = 0.0
    for i in range(len(p) - 1):
        total += float(haversine_nm(lats[i], lons[i], lats[i + 1], lons[i + 1]))
    return round(total, 1)


def _reference_build_actual_metrics(df_sample, actual_firs, actual_pts, filed_pts,
                                     eurocontrol_rates, ac_col):
    from flight_routes.costs import FUEL_KGH, JET_A_EUR_PER_KG, flight_atc_eur

    firs_by_id = dict(list(actual_firs.groupby("ECTRL ID")))
    pts_by_id = dict(list(actual_pts.groupby("ECTRL ID")))
    filed_by_id = dict(list(filed_pts.groupby("ECTRL ID")))
    meta = df_sample[["ECTRL ID", ac_col, "mtow_t"]].copy()
    meta["ECTRL ID"] = meta["ECTRL ID"].astype(str)

    records = []
    for _, mrow in meta.iterrows():
        eid = str(mrow["ECTRL ID"])
        if eid not in firs_by_id or eid not in pts_by_id:
            continue
        fir_dists, dur_h, total_dist = _ref_flight_actual_metrics(
            firs_by_id[eid], pts_by_id[eid], eurocontrol_rates
        )
        if not fir_dists or pd.isna(dur_h):
            continue
        ac, mtow = mrow[ac_col], mrow["mtow_t"]
        fkgh = FUEL_KGH.get(ac)
        rep = dict(fir_dists)
        rep["mtow_t"] = mtow
        atc_eur = flight_atc_eur(rep) if not pd.isna(mtow) else np.nan
        fuel_eur = round(fkgh * dur_h * JET_A_EUR_PER_KG, 2) if fkgh else np.nan
        fuel_kg = round(fkgh * dur_h, 1) if fkgh else np.nan
        cost_eur = round(atc_eur + fuel_eur, 2) if not (pd.isna(atc_eur) or pd.isna(fuel_eur)) else np.nan
        records.append({
            "ECTRL ID": int(eid),
            "actual_duration_h": round(dur_h, 4),
            "actual_total_dist_nm": round(total_dist, 1),
            "actual_atc_eur": atc_eur,
            "actual_fuel_eur": fuel_eur,
            "actual_fuel_kg": fuel_kg,
            "actual_cost_eur": cost_eur,
            "planned_dist_nm": _ref_total_dist_from_pts(filed_by_id[eid]) if eid in filed_by_id else np.nan,
        })
    return pd.DataFrame(records)


# --- synthetic multi-flight dataset covering the tricky cases ---

def _synthetic_dataset():
    # Flight 1: entirely inside one FIR
    # Flight 2: one segment crosses from FIRA into FIRB mid-segment (time-proportional split)
    # Flight 3: a point starts in a gap (no FIR covers it) before entering FIRA - that
    #           first segment's distance must be dropped, not assigned anywhere
    # Flight 4: crosses ABCDUIR, which must consolidate into ABCDFIR
    firs_rows = [
        # Flight 1
        dict(**{"ECTRL ID": 1, "FIR ID": "FIRA", "Entry Time": "01-09-2023 10:00:00", "Exit Time": "01-09-2023 12:00:00"}),
        # Flight 2
        dict(**{"ECTRL ID": 2, "FIR ID": "FIRA", "Entry Time": "01-09-2023 10:00:00", "Exit Time": "01-09-2023 11:00:00"}),
        dict(**{"ECTRL ID": 2, "FIR ID": "FIRB", "Entry Time": "01-09-2023 11:00:00", "Exit Time": "01-09-2023 12:00:00"}),
        # Flight 3
        dict(**{"ECTRL ID": 3, "FIR ID": "FIRA", "Entry Time": "01-09-2023 10:20:00", "Exit Time": "01-09-2023 12:00:00"}),
        # Flight 4
        dict(**{"ECTRL ID": 4, "FIR ID": "ABCDUIR", "Entry Time": "01-09-2023 10:00:00", "Exit Time": "01-09-2023 12:00:00"}),
    ]
    actual_firs = pd.DataFrame(firs_rows)
    actual_firs["ECTRL ID"] = actual_firs["ECTRL ID"].astype(str)

    pts_rows = [
        # Flight 1: 3 points, all within FIRA's window
        dict(**{"ECTRL ID": 1, "Time Over": "01-09-2023 10:00:00", "Latitude": 50.0, "Longitude": 0.0}),
        dict(**{"ECTRL ID": 1, "Time Over": "01-09-2023 10:30:00", "Latitude": 50.5, "Longitude": 0.5}),
        dict(**{"ECTRL ID": 1, "Time Over": "01-09-2023 11:00:00", "Latitude": 51.0, "Longitude": 1.0}),
        # Flight 2: segment from 10:50 (FIRA) to 11:10 (FIRB) straddles the 11:00 boundary
        dict(**{"ECTRL ID": 2, "Time Over": "01-09-2023 10:50:00", "Latitude": 50.0, "Longitude": 0.0}),
        dict(**{"ECTRL ID": 2, "Time Over": "01-09-2023 11:10:00", "Latitude": 51.0, "Longitude": 1.0}),
        # Flight 3: first point at 10:10 is BEFORE FIRA's entry (10:20) -> no FIR match (gap);
        # second point at 10:40 is inside FIRA. That first segment's distance is dropped.
        dict(**{"ECTRL ID": 3, "Time Over": "01-09-2023 10:10:00", "Latitude": 50.0, "Longitude": 0.0}),
        dict(**{"ECTRL ID": 3, "Time Over": "01-09-2023 10:40:00", "Latitude": 50.3, "Longitude": 0.3}),
        dict(**{"ECTRL ID": 3, "Time Over": "01-09-2023 11:00:00", "Latitude": 50.6, "Longitude": 0.6}),
        # Flight 4: 2 points inside the ABCDUIR window
        dict(**{"ECTRL ID": 4, "Time Over": "01-09-2023 10:00:00", "Latitude": 50.0, "Longitude": 0.0}),
        dict(**{"ECTRL ID": 4, "Time Over": "01-09-2023 11:00:00", "Latitude": 51.0, "Longitude": 1.0}),
    ]
    actual_pts = pd.DataFrame(pts_rows)
    actual_pts["ECTRL ID"] = actual_pts["ECTRL ID"].astype(str)

    filed_rows = []
    for eid in (1, 2, 3, 4):
        filed_rows += [
            dict(**{"ECTRL ID": eid, "Sequence Number": 1, "Latitude": 50.0, "Longitude": 0.0}),
            dict(**{"ECTRL ID": eid, "Sequence Number": 2, "Latitude": 50.5, "Longitude": 0.5}),
            dict(**{"ECTRL ID": eid, "Sequence Number": 3, "Latitude": 51.0, "Longitude": 1.0}),
        ]
    filed_pts = pd.DataFrame(filed_rows)
    filed_pts["ECTRL ID"] = filed_pts["ECTRL ID"].astype(str)

    df_sample = pd.DataFrame({
        "ECTRL ID": [1, 2, 3, 4],
        "AC Type": ["A320", "A320", "A320", "A320"],
        "mtow_t": [78.0, 78.0, 78.0, 78.0],
    })

    return df_sample, actual_firs, actual_pts, filed_pts


@pytest.mark.parametrize("ectrl_id", [1, 2, 3, 4])
def test_vectorised_matches_reference_per_flight(ectrl_id):
    df_sample, actual_firs, actual_pts, filed_pts = _synthetic_dataset()

    got = build_actual_metrics(df_sample, actual_firs, actual_pts, filed_pts,
                                EUROCONTROL_RATES, ac_col="AC Type")
    want = _reference_build_actual_metrics(df_sample, actual_firs, actual_pts, filed_pts,
                                           EUROCONTROL_RATES, ac_col="AC Type")

    got_row = got[got["ECTRL ID"] == ectrl_id].iloc[0]
    want_row = want[want["ECTRL ID"] == ectrl_id].iloc[0]

    for col in ["actual_duration_h", "actual_total_dist_nm", "actual_atc_eur",
                "actual_fuel_eur", "actual_fuel_kg", "actual_cost_eur", "planned_dist_nm"]:
        assert got_row[col] == pytest.approx(want_row[col], abs=0.05), (
            f"flight {ectrl_id}, column {col}: vectorised={got_row[col]!r} vs reference={want_row[col]!r}"
        )


def test_flight_3_gap_segment_matches_reference_exactly():
    """Flight 3's first point (10:10) is before FIRA's entry (10:20), so it's outside the
    airborne window and filtered out entirely by both implementations before any FIR
    matching happens - only one real segment (10:40->11:00, inside FIRA) remains. This is
    already covered by the parametrized test above; this just makes the reasoning explicit
    so a future reader doesn't assume there are two segments here."""
    df_sample, actual_firs, actual_pts, filed_pts = _synthetic_dataset()
    got = build_actual_metrics(df_sample, actual_firs, actual_pts, filed_pts,
                                EUROCONTROL_RATES, ac_col="AC Type")
    row = got[got["ECTRL ID"] == 3].iloc[0]
    fira_seg = float(haversine_nm(50.3, 0.3, 50.6, 0.6))
    assert row["actual_total_dist_nm"] == pytest.approx(fira_seg, abs=0.05)


def test_remap_uir_to_fir():
    """Direct unit test of the consolidation rule, isolated from flight_atc_eur (which
    always reads costs.EUROCONTROL_RATES directly, not whatever dict is passed here -
    testing consolidation through the cost output would be confounded by that)."""
    from flight_routes.variance import _remap_uir_to_fir

    # real codes are a 4-letter prefix plus a FIR/UIR suffix that replaces, not appends
    # (e.g. EGTTUIR -> EGTT + FIR = EGTTFIR), so the fake codes here follow that shape
    rates = {"ABCDFIR": 70.0}
    assert _remap_uir_to_fir("ABCDUIR", rates) == "ABCDFIR"  # base has a rate: consolidate
    assert _remap_uir_to_fir("WXYZUIR", rates) == "WXYZUIR"  # base has no rate: leave as-is
    assert _remap_uir_to_fir("ABCDFIR", rates) == "ABCDFIR"  # not a UIR: untouched
