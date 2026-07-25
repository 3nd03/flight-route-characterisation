"""FIR-crossing feature detection and binary route signatures.

Column filter is on name suffix (FIR/UIR), not dtype == float64 - the dtype
filter also matches ADEP/ADES Latitude/Longitude and Requested FL, which
corrupts both the clustering distance metric and the cluster-merge step.
"""


def fir_columns(df):
    """All FIR/UIR distance columns in a joined FIR+Flights DataFrame."""
    return [c for c in df.columns if c.endswith(("FIR", "UIR"))]


def active_firs_for_group(group_df, fir_cols, threshold=0.25):
    """FIRs crossed by at least `threshold` fraction of flights in one O-D pair group."""
    return [f for f in fir_cols if (group_df[f] > 0).mean() >= threshold]


def detect_active_firs(df, fir_cols, threshold=0.25):
    """Union, across all O-D pairs in df, of each pair's active FIRs."""
    active = set()
    for _, group_df in df.groupby(["ADEP", "ADES"]):
        active.update(active_firs_for_group(group_df, fir_cols, threshold))
    return sorted(active)


def make_route_signatures(df, fir_cols):
    """Binary route signature: 1 if a flight crossed a given FIR, 0 if not."""
    return (df[fir_cols] > 0).astype(int)
