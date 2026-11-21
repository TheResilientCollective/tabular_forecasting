# Notebooks

All notebooks are [marimo](https://marimo.io) notebooks stored as plain Python files.

## Setup

From the project root, install the dependencies once:

```bash
uv sync
```

## Start the marimo server

Run these from the project root.

Open the notebook browser for the whole folder:

```bash
uv run marimo edit notebooks
```

Open a single notebook for editing:

```bash
uv run marimo edit notebooks/ili_case/review_covid.py
```

Serve a notebook as a read-only app (code hidden):

```bash
uv run marimo run notebooks/ili_case/review_covid.py
```

marimo prints the URL (by default `http://localhost:2718`) and opens it in the browser.
Useful options: `--port 2719` to pick another port, `--headless` to skip opening the browser.
Stop the server with `Ctrl+C`.

## Notebooks

The ILI notebooks are in `notebooks/ili_case/` and the mpox notebooks are in `notebooks/mpox/`.

| Notebook | Purpose | Runs models |
| --- | --- | --- |
| `ili_case/review_covid.py` | Review saved COVID results, including Rt (needs `notebooks/ili_case/rt_ili.py` outputs) and hospital admissions (needs `notebooks/ili_case/ili_hosp.py` outputs) | no |
| `ili_case/review_flu.py` | Review saved flu results, including Rt (needs `notebooks/ili_case/rt_ili.py` outputs) and hospital admissions (needs `notebooks/ili_case/ili_hosp.py` outputs) | no |
| `ili_case/review_rsv.py` | Review saved RSV results, including Rt (needs `notebooks/ili_case/rt_ili.py` outputs) and hospital admissions (needs `notebooks/ili_case/ili_hosp.py` outputs) | no |
| `mpox/review_mpox.py` | Review saved mpox results | no |
| `mpox/review_mpox_regional.py` | Review saved regional mpox results (San Diego, Los Angeles, San Francisco, New York City, California) | no |
| `ili_case/ili_cases.py` | Backtest and forecast COVID, flu and RSV with Timestral, TimesFM 3.0 and TabPFN-TS | yes |
| `mpox/mpox.py` | Backtest and forecast mpox with Timestral, TimesFM 3.0 and TabPFN-TS (about an hour on CPU) | yes |
| `mpox/mpox_regional.py` | Backtest and forecast the five regional mpox series with Timestral, TimesFM 3.0 and TabPFN-TS | yes |
| `ili_case/rt_ili.py` | Rt for COVID, flu and RSV: renewal estimate, TabPFN-TS Rt forecast, renewal fit with TabPFN Bayesian optimisation | yes |
| `ili_case/ili_hosp.py` | Backtest and forecast ILI hospital admissions (COVID, flu, RSV) with Timestral, TimesFM 3.0, TabPFN-TS and naive | yes |

The review notebooks only read the CSVs in `data/*/forecast_tabular/`, so they start quickly
and need no model download.

The run notebooks download the Timestral and TimesFM checkpoints from Hugging Face on first
use and run inference on CPU. marimo executes a notebook's cells as soon as it is opened, so
opening `ili_case/ili_cases.py`, `ili_case/ili_hosp.py`, `ili_case/rt_ili.py`, `mpox/mpox.py` or `mpox/mpox_regional.py` starts the models and overwrites the saved CSVs.

TabPFN-TS and the Bayesian-optimisation step use the local TabPFN-3.5 weights, which are under a
non-commercial license. Accept the license once at https://ux.priorlabs.ai and provide your API
key as `TABPFN_TOKEN`, either exported in the shell or as a `TABPFN_TOKEN=...` line in a `.env`
file in the project root (keep that file out of version control).

The mpox notebooks forecast one month out: a weekly path through the original target week plus
three more weeks, saved as `mpox_forecast_path.csv` and `regional_forecast_path.csv`. The
original VAR_LASSO forecast only covers the target week.

## Run without the server

Any notebook can also be executed as a script:

```bash
uv run python notebooks/ili_case/ili_cases.py
```
