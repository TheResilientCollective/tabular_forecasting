import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(ROOT))

    import forecasting as fc

    return ROOT, fc, mo, np, pd, plt


@app.cell
def _(mo):
    mo.md(
        """
# ILI hospital admissions: zero-shot forecasts with Timestral, TimesFM 3.0 and TabPFN-TS

Univariate, zero-shot forecasts of weekly COVID, flu and RSV hospital admissions.
The series is the `Observations` column of the original hospital reports, which holds
18 observed weeks per disease (2026-07-18 to 2026-11-14). The original file also has
a three-week forecast, so the models forecast four weeks ahead to match the one-month
horizon of the case forecasts.

The series is short, so the rolling-origin backtest uses a minimum context of 6 weeks.
That gives three origins per disease, so the backtest scores are indicative only.
The most recent weeks are partial-data estimates and may be revised upward.
TabPFN-TS runs the local TabPFN-3.5 weights and needs `TABPFN_TOKEN` (environment or `.env`).
The seasonal-naive baseline is left out, because it falls back to naive on a series this short.
"""
    )
    return


@app.cell
def _(ROOT, pd):
    DATA = ROOT / "data" / "ili_cases"
    DISEASES = ["COVID", "FLU", "RSV"]

    history = {}
    original = {}
    for _disease in DISEASES:
        _hosp = pd.read_csv(DATA / "forecast_original" / f"{_disease}_hosp_reports.csv", parse_dates=["date"])
        _observed = _hosp[_hosp["Observations"].notna()]
        history[_disease] = _observed.set_index("date")["Observations"].astype(float)
        original[_disease] = _hosp

    original_weeks = {
        _disease: original[_disease].loc[original[_disease]["type"] == "forecast", "date"].tolist()
        for _disease in DISEASES
    }
    horizon = 4
    last_week = {_disease: history[_disease].index.max() for _disease in DISEASES}
    future_weeks = {
        _disease: [last_week[_disease] + pd.Timedelta(weeks=_step) for _step in range(1, horizon + 1)]
        for _disease in DISEASES
    }
    assert all(_weeks[:len(original_weeks[_disease])] == original_weeks[_disease] for _disease, _weeks in future_weeks.items())
    return DATA, DISEASES, future_weeks, history, horizon, last_week, original


@app.cell
def _(fc):
    timestral_model = fc.load_timestral()
    timesfm_evaluator, timesfm_levels = fc.load_timesfm()
    predictors = {
        "timestral": fc.timestral_predictor(timestral_model),
        "timesfm": fc.timesfm_predictor(timesfm_evaluator, timesfm_levels),
        "tabpfn_ts": fc.tabpfn_ts_predictor(fc.load_tabpfn_ts()),
        "naive": fc.naive_predictor,
    }
    return (predictors,)


@app.cell
def _(DISEASES, ROOT, fc, history, horizon, predictors):
    backtest_table = fc.backtest(
        {_disease: history[_disease].to_numpy() for _disease in DISEASES},
        horizon=horizon,
        predictors=predictors,
        n_windows=20,
        min_context=6,
        weeks={_disease: history[_disease].index for _disease in DISEASES},
    )
    backtest_summary = fc.summarise(backtest_table)
    backtest_table.to_csv(ROOT / "data" / "ili_cases" / "forecast_tabular" / "hosp_backtest_windows.csv", index=False)
    backtest_summary
    return


@app.cell
def _(DISEASES, ROOT, future_weeks, fc, history, horizon, last_week, pd, predictors):
    arrays = [history[_disease].to_numpy() for _disease in DISEASES]
    ends = [last_week[_disease] for _disease in DISEASES]

    forecast_rows = []
    for _name, _predict in predictors.items():
        _outputs = fc.predict_dated(_predict, arrays, horizon, ends)
        for _disease, _out in zip(DISEASES, _outputs):
            for _step in range(horizon):
                _row = {
                    "disease": _disease,
                    "model": _name,
                    "week_end": future_weeks[_disease][_step].strftime("%Y-%m-%d"),
                    "horizon": _step + 1,
                    "median": float(_out.median[_step]),
                }
                if _out.quantiles is not None:
                    for _label, _lower, _upper in [("80", 0.1, 0.9), ("90", 0.05, 0.95)]:
                        _low, _high = fc.interval(_out, _lower, _upper)
                        _row[f"lower_{_label}"] = float(_low[_step])
                        _row[f"upper_{_label}"] = float(_high[_step])
                forecast_rows.append(_row)

    forecast_table = pd.DataFrame(forecast_rows)
    forecast_table.to_csv(ROOT / "data" / "ili_cases" / "forecast_tabular" / "ili_hosp_forecast.csv", index=False)
    forecast_table
    return (forecast_table,)


@app.cell
def _(DISEASES, forecast_table, history, last_week, original, pd, plt):
    fig, axes = plt.subplots(len(DISEASES), 2, figsize=(14, 12))
    for _row, _disease in zip(axes, DISEASES):
        _orig = original[_disease]
        _future = _orig[_orig["type"] == "forecast"]
        for _ax, _window in zip(_row, ["entire", "last 60 days"]):
            _observed = history[_disease]
            if _window == "last 60 days":
                _observed = _observed[_observed.index >= last_week[_disease] - pd.Timedelta(days=60)]
            _ax.plot(_observed.index, _observed.values, color="black", marker="o", label="observed")
            _ax.plot(_future["date"], _future["median"], "k--", label="original median")
            _ax.fill_between(
                _future["date"], _future["lower_90"], _future["upper_90"], color="grey", alpha=0.3,
                label="original 90%",
            )
            for _model, _color in [("timestral", "tab:blue"), ("timesfm", "tab:orange"), ("tabpfn_ts", "tab:green")]:
                _f = forecast_table[(forecast_table["disease"] == _disease) & (forecast_table["model"] == _model)]
                _x = pd.to_datetime(_f["week_end"])
                _ax.plot(_x, _f["median"], color=_color, marker="o", label=_model)
                _ax.fill_between(_x, _f["lower_90"], _f["upper_90"], color=_color, alpha=0.15)
            _ax.set_title(f"{_disease} hospital admissions: {_window}")
            _ax.set_ylabel("admissions")
            _ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig
    return


if __name__ == "__main__":
    app.run()
