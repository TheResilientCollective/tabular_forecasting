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
# ILI cases: zero-shot forecasts with Timestral, TimesFM 3.0 and TabPFN-TS

Univariate, zero-shot forecasts of weekly COVID, flu and RSV case counts (San Diego).
TabPFN-TS runs the local TabPFN-3.5 weights and needs `TABPFN_TOKEN` (environment or `.env`).
Each model is scored on a rolling-origin backtest over the history against naive and
seasonal-naive baselines. The models forecast one month ahead (four weeks), which extends
the three weeks of the original forecast by one week, and the forecasts are overlaid on it.
The original forecast covers a single origin, so it is compared visually rather than scored.
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
        _inp = pd.read_csv(DATA / "input" / f"{_disease}.csv", parse_dates=["date_week_end"])
        history[_disease] = _inp.set_index("date_week_end")["Cases"].astype(float)
        original[_disease] = pd.read_csv(
            DATA / "forecast_original" / f"{_disease}_case_reports.csv", parse_dates=["epiweek_end"]
        )

    original_weeks = {
        _disease: original[_disease].loc[original[_disease]["Type"] == "forecast", "epiweek_end"].tolist()
        for _disease in DISEASES
    }
    horizon = 4
    last_week = history["COVID"].index.max()
    future_weeks = {
        _disease: [last_week + pd.Timedelta(weeks=_step) for _step in range(1, horizon + 1)]
        for _disease in DISEASES
    }
    assert all(_weeks[:len(original_weeks[_disease])] == original_weeks[_disease] for _disease, _weeks in future_weeks.items())
    return DISEASES, future_weeks, history, horizon, original


@app.cell
def _(fc):
    timestral_model = fc.load_timestral()
    timesfm_evaluator, timesfm_levels = fc.load_timesfm()
    predictors = {
        "timestral": fc.timestral_predictor(timestral_model),
        "timesfm": fc.timesfm_predictor(timesfm_evaluator, timesfm_levels),
        "tabpfn_ts": fc.tabpfn_ts_predictor(fc.load_tabpfn_ts()),
        "naive": fc.naive_predictor,
        "seasonal_naive": fc.seasonal_naive_predictor,
    }
    return (predictors,)


@app.cell
def _(DISEASES, ROOT, fc, history, horizon, predictors):
    backtest_table = fc.backtest(
        {_disease: history[_disease].to_numpy() for _disease in DISEASES},
        horizon=horizon,
        predictors=predictors,
        n_windows=20,
        min_context=52,
        weeks={_disease: history[_disease].index for _disease in DISEASES},
    )
    backtest_summary = fc.summarise(backtest_table)
    backtest_table.to_csv(ROOT / "data" / "ili_cases" / "forecast_tabular" / "backtest_windows.csv", index=False)
    backtest_summary
    return


@app.cell
def _(DISEASES, ROOT, future_weeks, fc, history, horizon, pd, predictors):
    arrays = [history[_disease].to_numpy() for _disease in DISEASES]
    week_labels = [_week.strftime("%Y-%m-%d") for _week in future_weeks["COVID"]]

    forecast_rows = []
    for _name, _predict in predictors.items():
        _outputs = fc.predict_dated(_predict, arrays, horizon, [history[_disease].index.max() for _disease in DISEASES])
        for _disease, _out in zip(DISEASES, _outputs):
            for _step in range(horizon):
                _row = {
                    "disease": _disease,
                    "model": _name,
                    "week_end": week_labels[_step],
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
    forecast_table.to_csv(ROOT / "data" / "ili_cases" / "forecast_tabular" / "ili_cases_forecast.csv", index=False)
    forecast_table
    return (forecast_table,)


@app.cell
def _(DISEASES, forecast_table, history, original, pd, plt):
    fig, axes = plt.subplots(len(DISEASES), 1, figsize=(10, 12))
    for _ax, _disease in zip(axes, DISEASES):
        _recent = history[_disease].iloc[-52:]
        _ax.plot(_recent.index, _recent.values, color="black", label="observed")

        _orig = original[_disease]
        _future = _orig[_orig["Type"] == "forecast"]
        _ax.plot(_future["epiweek_end"], _future["Median"], "k--", label="original median")
        _ax.fill_between(
            _future["epiweek_end"], _future["Lower90"], _future["Upper90"], color="grey", alpha=0.3,
            label="original 90%",
        )

        for _model, _color in [("timestral", "tab:blue"), ("timesfm", "tab:orange"), ("tabpfn_ts", "tab:green")]:
            _f = forecast_table[(forecast_table["disease"] == _disease) & (forecast_table["model"] == _model)]
            _x = pd.to_datetime(_f["week_end"])
            _ax.plot(_x, _f["median"], color=_color, marker="o", label=_model)
            _ax.fill_between(_x, _f["lower_90"], _f["upper_90"], color=_color, alpha=0.15)

        _ax.set_title(_disease)
        _ax.set_ylabel("cases")
        _ax.legend(loc="upper left")
    fig.tight_layout()
    fig
    return


if __name__ == "__main__":
    app.run()
