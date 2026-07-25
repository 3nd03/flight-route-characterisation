import math

from flight_routes.costs import EUROCONTROL_RATES, flight_atc_eur, flight_fuel_eur, _parse_duration


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
