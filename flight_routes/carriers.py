"""Airline operator classification: low-cost vs full-service carrier.

STATFOR Market Segment - the EUROCONTROL column meant for exactly this -
is "Not Classified" for all 899,495 rows of the Sep 2023 export, unusable.
AC Operator (555 unique ICAO-style operator codes) is populated and does
distinguish real carriers, but has no business-model label attached, so
this is a hand-curated lookup, same pattern as costs.MTOW_TONNES/FUEL_KGH:
it only covers operators common and unambiguous enough to classify with
confidence, not all 555. Best-effort, not sourced from an official
low-cost/full-service registry - spot check before citing in the report.

summarise_self_deltas_by_carrier() reuses variance.compute_self_deltas'
output (the planned-vs-realised deltas), just regrouped by carrier_type
instead of O-D pair/cluster.

"ZZZ" is almost certainly a placeholder for unidentified or general-aviation
operators, not a real airline - it is deliberately left out of both
categories and mapped to "unclassified" rather than guessed into either
bucket. Its share depends on which denominator: 164,807 of the raw
899,495-row Sep 2023 export (~18.3%), but 79,193 of the ~536,520-flight
qualifying-O-D-pair subset used for the full-dataset analysis (~14.8%) -
the qualifying-pairs filter (min_flights_per_od) disproportionately drops
one-off ZZZ flights along with everything else below the threshold.

Coverage on the full qualifying-pairs dataset (measured 2026-07-27):
LOW_COST 18.4%, FULL_SERVICE 34.5%, unclassified 47.1% before the second
round of codes below was added (PGT/TVF/AEE/LOT/AEA/QTR/BEL/UAE/EIN,
identified from the actual top-30 AC Operator counts on that dataset,
recovering roughly a further 10 percentage points). Genuinely ambiguous
codes seen in that same top-30 were deliberately left unclassified rather
than guessed: WIF (Wideroe, Norwegian regional feeder), SXS (SunExpress,
a Lufthansa/Turkish Airlines leisure-charter joint venture), ANE (Air
Nostrum, Iberia's regional feeder) - none of these fit a clean low-cost/
full-service split.
"""

from .variance import DELTA_COLS_SELF, _summarise

LOW_COST = {
    "RYR",  # Ryanair
    "EZY",  # easyJet
    "VLG",  # Vueling
    "EWG",  # Eurowings
    "WZZ",  # Wizz Air
    "TRA",  # Transavia
    "VOE",  # Volotea
    "EXS",  # Jet2.com
    "PGT",  # Pegasus Airlines
    "TVF",  # Transavia France (distinct ICAO code from parent TRA)
}

FULL_SERVICE = {
    "THY",  # Turkish Airlines
    "DLH",  # Lufthansa
    "AFR",  # Air France
    "SAS",  # SAS Scandinavian Airlines
    "KLM",  # KLM
    "BAW",  # British Airways
    "IBE",  # Iberia
    "SWR",  # Swiss International Air Lines
    "AUA",  # Austrian Airlines
    "TAP",  # TAP Air Portugal
    "FIN",  # Finnair
    "AEE",  # Aegean Airlines
    "LOT",  # LOT Polish Airlines
    "AEA",  # Air Europa
    "QTR",  # Qatar Airways
    "BEL",  # Brussels Airlines
    "UAE",  # Emirates
    "EIN",  # Aer Lingus - borderline: IAG flag carrier, but has budget-airline
            # characteristics on short-haul; classified full-service on balance
}


def carrier_type(ac_operator):
    """'low_cost', 'full_service', or 'unclassified' for one AC Operator code."""
    if ac_operator in LOW_COST:
        return "low_cost"
    if ac_operator in FULL_SERVICE:
        return "full_service"
    return "unclassified"


def add_carrier_type(df, operator_col="AC Operator"):
    """Add a carrier_type column ('low_cost'/'full_service'/'unclassified'). Returns a new DataFrame."""
    df = df.copy()
    df["carrier_type"] = df[operator_col].map(carrier_type)
    return df


def carrier_coverage(df, operator_col="AC Operator"):
    """Fraction of flights falling into each carrier_type - report this alongside
    any carrier-type breakdown, since 'unclassified' can be a large share (see module docstring)."""
    return (
        df[operator_col].map(carrier_type)
        .value_counts(normalize=True)
        .rename("fraction")
        .rename_axis("carrier_type")
        .reset_index()
    )


def summarise_self_deltas_by_carrier(df_compared, operator_col="AC Operator"):
    """Type 2 self-delta summary (see variance.compute_self_deltas), grouped by
    carrier_type instead of O-D pair/cluster. df_compared must already have the
    delta_*_self columns, i.e. compute_self_deltas must have been called first."""
    df_compared = add_carrier_type(df_compared, operator_col)
    return _summarise(df_compared, ["carrier_type"], DELTA_COLS_SELF)
