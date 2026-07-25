# flight-route-characterisation

Data-driven characterisation of flight route alternatives and their cost, built as an input layer for Mercury, the air transport mobility simulator developed at the University of Westminster's Centre for Air Traffic Management Research. Given an aircraft type and origin-destination pair, it returns the historically observed route alternatives, ranked by predicted cost or duration, instead of assuming a single planned route.

Built under the University of Westminster "Students as Co-Creators" programme. Student partner: Óscar Denche Morant. Academic partners: Luis Delgado, Michal Weiszer.

## Quick start

```bash
git clone https://github.com/3nd03/flight-route-characterisation.git
cd flight-route-characterisation
pip install -e .
```

```python
from flight_routes.query import predict_route_options
from flight_routes.data import cache_dir
import pandas as pd

full_summary    = pd.read_csv(cache_dir() / "full_summary.csv")
full_summary_ac = pd.read_csv(cache_dir() / "full_summary_ac.csv")

predict_route_options("A320", "LEBL", "LPPT", full_summary, full_summary_ac)
```

`full_summary`/`full_summary_ac` ship in the repo (`data/processed/`) — no raw data needed just to query. They cover 4,558 O-D pairs (536,520 flights, September 2023), of which 2,494 (54.7%) show genuine route or cost alternatives.

## What it does

1. Groups historical flights by O-D pair, then splits them by which flight information regions (FIRs) they crossed, then by the distance flown within each FIR — a three-layer clustering that separates *which corridor* a flight took from *how it flew within that corridor*.
2. Attaches a deterministic cost model (EUROCONTROL en-route charges + Nav Canada + fuel) to each resulting cluster.
3. Compares each cluster's realised flights against two different reference points: the cluster's own planned representative (the noise a simulator should sample around it), and each flight's own filed plan (a measure of route-planning reliability, independent of clustering).

Full methodology, validation, and findings: [`docs/report_draft.md`](docs/report_draft.md) (the DRC report) and [`docs/technical_documentation.md`](docs/technical_documentation.md).

## Repo layout

```
flight_routes/       the library
    data.py           loading, joining, caching (raw + processed paths configurable)
    features.py       FIR-crossing detection, route signatures
    clustering.py     three-layer clustering
    costs.py          ATC/fuel cost model
    variance.py       planned-vs-realised comparisons
    validation.py     cluster quality checks, outlier handling
    query.py          predict_route_options() / query_route_profile() - the public API
    plotting.py       all figures
notebooks/          Colab orchestration notebook + a local route-map viewer
data/
    raw/              large EUROCONTROL source files - not tracked in git
    processed/         small aggregated outputs (full_summary etc.) - tracked in git
docs/                DRC report, technical write-up, images
tests/               unit tests (pytest)
```

## Configuring where the raw data lives

`flight_routes.data` reads two environment variables, each defaulting to `data/{raw,processed}` inside the repo:

- `FLIGHT_ROUTES_RAW_DIR` — the large EUROCONTROL parquet/csv exports (only needed to rebuild `full_summary` from scratch, e.g. on new historical data)
- `FLIGHT_ROUTES_CACHE_DIR` — where computed outputs are cached

Set these any time before calling a `flight_routes.data` function, e.g. after mounting Google Drive in Colab:

```python
import os
os.environ["FLIGHT_ROUTES_RAW_DIR"] = "/drive/MyDrive/flight-project"
```

## Tests

```bash
pytest tests/
```
