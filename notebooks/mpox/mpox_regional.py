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
    import review as rv

    return ROOT, fc, mo, np, pd, plt, rv


@app.cell
def _(mo):
    mo.md("""
    # Mpox, regional: San Diego, Los Angeles, San Francisco, New York City, California

    Zero-shot forecasts with Timestral, TimesFM 3.0 and TabPFN-TS for five regions, with naive and
    seasonal-naive baselines. TabPFN-TS runs the local TabPFN-3.5 weights and needs `TABPFN_TOKEN`
    (environment or `.env`).

    The input file `mpox_aggregated_geo.csv` only has states, so the city and county series are
    taken from the observation rows of the original forecast file `mpox_case_reports.csv`
    (California is identical in both files).

    Two parts:

    - **Backtest**: rolling-origin windows per region at 1-week and 4-week horizons.
    - **Forecast** to the target week of the original forecast. Only observations before the
      target week are used. San Diego and San Francisco already have an observed target week,
      which is held out and used to score the forecasts; it is a single and possibly preliminary
      week per region. Los Angeles data stops 12 weeks before the target, so its forecast is a
      12-week-ahead one.
    - **One month out**: the model forecasts continue for three weeks past the target week, giving
      a weekly path. The original forecast only covers the target week, so the later weeks have no
      comparison; the 4-week backtest is the evidence for that range.
    """)
    return


@app.cell
def _(pd, rv):
    observed = rv.mpox_regional_history()
    original = rv.mpox_original_forecast()
    target_week = original["epiweek_start"].max()

    series = {}
    held_out = {}
    for _region, _key in rv.MPOX_REGIONS.items():
        _values = observed[observed["jurisdiction"] == _key].set_index("epiweek_start")["cases"].astype(float)
        assert not _values.empty and _values.notna().all()
        assert (_values.index.to_series().diff().dropna() == pd.Timedelta(weeks=1)).all()
        held_out[_region] = float(_values[target_week]) if target_week in _values.index else None
        series[_region] = _values[_values.index < target_week]

    horizons = {_region: (target_week - _values.index.max()).days // 7 for _region, _values in series.items()}
    overview = pd.DataFrame(
        {
            "weeks": {_r: len(_v) for _r, _v in series.items()},
            "first_week": {_r: _v.index.min().date() for _r, _v in series.items()},
            "last_context_week": {_r: _v.index.max().date() for _r, _v in series.items()},
            "forecast_horizon": horizons,
            "zero_share": {_r: float((_v == 0).mean()) for _r, _v in series.items()},
            "mean_last_12": {_r: float(_v.tail(12).mean()) for _r, _v in series.items()},
            "observed_target_week": held_out,
        }
    )
    overview
    return held_out, horizons, original, series, target_week


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
def _(fc, pd, predictors, rv, series):
    _tables = []
    for _horizon in [1, 4]:
        _table = fc.backtest(
            {_region: _values.to_numpy() for _region, _values in series.items()},
            horizon=_horizon,
            predictors=predictors,
            n_windows=20,
            min_context=52,
            weeks={_region: _values.index for _region, _values in series.items()},
        )
        _table.insert(1, "horizon", _horizon)
        _tables.append(_table)
    backtest_table = pd.concat(_tables, ignore_index=True)
    backtest_table.to_csv(rv.MPOX_DIR / "forecast_tabular" / "regional_backtest_windows.csv", index=False)
    rv.mpox_regional_summary(backtest_table)
    return (backtest_table,)


@app.cell
def _(backtest_table, rv):
    rv.mpox_regional_pivot(backtest_table, "mase")
    return


@app.cell
def _(fc, held_out, horizons, original, pd, predictors, rv, series, target_week):
    _rows = []
    _path_rows = []
    for _region, _values in series.items():
        _horizon = horizons[_region]
        _orig = original[original["jurisdiction"] == rv.MPOX_REGIONS[_region]].set_index("method")["predicted"]
        for _model, _predict in predictors.items():
            _out = fc.predict_dated(_predict, [_values.to_numpy()], _horizon + 3, [_values.index.max()])[0]
            _path_rows.extend(fc.forecast_path_rows(_out, _values.index.max(), target_week, region=_region, model=_model))
            _row = {
                "region": _region,
                "model": _model,
                "last_context_week": _values.index.max().strftime("%Y-%m-%d"),
                "target_week": target_week.strftime("%Y-%m-%d"),
                "horizon": _horizon,
                "median": float(_out.median[_horizon - 1]),
            }
            for _label, _lower, _upper in [("80", 0.1, 0.9), ("90", 0.05, 0.95)]:
                _low, _high = fc.interval(_out, _lower, _upper)
                _row[f"lower_{_label}"] = float(_low[_horizon - 1])
                _row[f"upper_{_label}"] = float(_high[_horizon - 1])
            _row["observed"] = held_out[_region]
            for _method in ["NAIVE", "AR_LASSO", "VAR_LASSO"]:
                _row[f"original_{_method}"] = float(_orig[_method]) if _method in _orig.index else None
            _rows.append(_row)

    forecast_table = pd.DataFrame(_rows)
    forecast_table.to_csv(rv.MPOX_DIR / "forecast_tabular" / "regional_forecast.csv", index=False)
    forecast_path = pd.DataFrame(_path_rows)
    forecast_path.to_csv(rv.MPOX_DIR / "forecast_tabular" / "regional_forecast_path.csv", index=False)
    forecast_table
    return forecast_path, forecast_table


@app.cell
def _(forecast_table, rv):
    rv.mpox_regional_target_errors(forecast_table)
    return


@app.cell
def _(forecast_path, forecast_table, plt, rv, series):
    fig, axes = plt.subplots(len(series), 1, figsize=(10, 3.2 * len(series)))
    for _ax, (_region, _values) in zip(axes, series.items()):
        _first = forecast_table[forecast_table["region"] == _region].iloc[0]
        rv.plot_mpox_path(
            _ax,
            _region,
            _values,
            forecast_path[forecast_path["region"] == _region],
            target_week=_first["target_week"],
            original=_first["original_VAR_LASSO"],
            observed=_first["observed"],
        )
    fig.tight_layout()
    fig
    return


if __name__ == "__main__":
    app.run()
