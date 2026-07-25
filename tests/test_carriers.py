import pandas as pd

from flight_routes.carriers import add_carrier_type, carrier_coverage, carrier_type, summarise_self_deltas_by_carrier


def test_known_low_cost_codes():
    assert carrier_type("RYR") == "low_cost"
    assert carrier_type("EZY") == "low_cost"
    assert carrier_type("VLG") == "low_cost"


def test_known_full_service_codes():
    assert carrier_type("THY") == "full_service"
    assert carrier_type("DLH") == "full_service"
    assert carrier_type("KLM") == "full_service"


def test_second_round_codes_added_from_real_top30_counts():
    # identified from df_full's real AC Operator top-30 counts (2026-07-27)
    assert carrier_type("PGT") == "low_cost"       # Pegasus Airlines
    assert carrier_type("TVF") == "low_cost"       # Transavia France
    assert carrier_type("AEE") == "full_service"   # Aegean Airlines
    assert carrier_type("LOT") == "full_service"   # LOT Polish Airlines
    assert carrier_type("AEA") == "full_service"   # Air Europa
    assert carrier_type("QTR") == "full_service"   # Qatar Airways
    assert carrier_type("BEL") == "full_service"   # Brussels Airlines
    assert carrier_type("UAE") == "full_service"   # Emirates
    assert carrier_type("EIN") == "full_service"   # Aer Lingus (borderline, see module docstring)


def test_ambiguous_business_model_codes_left_unclassified():
    # deliberately not guessed into either bucket - see module docstring
    assert carrier_type("WIF") == "unclassified"  # Wideroe, regional feeder
    assert carrier_type("SXS") == "unclassified"  # SunExpress, leisure JV
    assert carrier_type("ANE") == "unclassified"  # Air Nostrum, regional feeder


def test_zzz_placeholder_and_unknown_codes_are_unclassified():
    assert carrier_type("ZZZ") == "unclassified"
    assert carrier_type("XYZ123") == "unclassified"


def test_add_carrier_type_adds_column_without_mutating_input():
    df = pd.DataFrame({"AC Operator": ["RYR", "THY", "ZZZ"]})
    out = add_carrier_type(df)
    assert list(out["carrier_type"]) == ["low_cost", "full_service", "unclassified"]
    assert "carrier_type" not in df.columns


def test_carrier_coverage_fractions_sum_to_one():
    df = pd.DataFrame({"AC Operator": ["RYR", "RYR", "THY", "ZZZ"]})
    cov = carrier_coverage(df)
    assert abs(cov["fraction"].sum() - 1.0) < 1e-9
    row = cov.set_index("carrier_type")
    assert row.loc["low_cost", "fraction"] == 0.5


def test_summarise_self_deltas_by_carrier_groups_correctly():
    df_compared = pd.DataFrame({
        "AC Operator":            ["RYR", "RYR", "THY", "ZZZ"],
        "delta_dist_nm_self":     [1.0, 3.0, 10.0, 5.0],
        "delta_duration_h_self":  [0.1, 0.3, 0.2, 0.1],
        "delta_fuel_kg_self":     [10.0, 30.0, 20.0, 10.0],
    })
    out = summarise_self_deltas_by_carrier(df_compared).set_index("carrier_type")
    assert out.loc["low_cost", "n"] == 2
    assert out.loc["low_cost", "delta_dist_nm_self_mean"] == 2.0
    assert out.loc["full_service", "n"] == 1
    assert out.loc["unclassified", "n"] == 1
