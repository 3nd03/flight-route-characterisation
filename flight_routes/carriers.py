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

"ZZZ" (164,807 flights, ~18% of the whole Sep 2023 dataset, by far the
single largest operator code) is almost certainly a placeholder for
unidentified or general-aviation operators, not a real airline. It is
deliberately left out of both categories and mapped to "unclassified"
rather than guessed into either bucket.
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
