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

    return ROOT, fc, mo, pd, plt, rv


@app.cell
def _(mo):
    mo.md("""
    # Mpox: zero-shot forecasts with Timestral, TimesFM 3.0 and TabPFN-TS

    Weekly mpox case counts per US jurisdiction, forecast univariately up to the target week of
    the original forecast (NAIVE, AR_LASSO, VAR_LASSO). Series end on different weeks, so the
    horizon is one week for most jurisdictions and two for California. Models are scored on a
    rolling-origin backtest per horizon. The original forecast has no observed target week, so
    the comparison with it measures agreement only, not accuracy, and covers the jurisdictions
    present in both.

    The foundation-model forecasts are extended one month out: a weekly path that runs through
    the target week plus three further weeks. VAR_LASSO only covers the target week, so the later
    weeks have nothing to be compared with; a separate 4-week backtest shows how the models do at
    that range.

    Recent weeks carry `Preliminary_*` week types and are subject to revision; they are used as-is.
    Most series are near zero, so MASE is undefined for some series and is reported as a count.

    TabPFN-TS runs the local TabPFN-3.5 weights and needs `TABPFN_TOKEN` (environment or `.env`).
    It takes a few seconds per backtest window, so this notebook runs for the better part of an hour.
    """)
    return


@app.cell
def _(ROOT, pd):
    MPOX = ROOT / "data" / "mpox"
    history_raw = pd.read_csv(
        MPOX / "input" / "mpox_aggregated_geo.csv",
        usecols=["Jurisdiction", "date_week_start", "Cases", "Week_Type"],
        parse_dates=["date_week_start"],
    )
    original = pd.read_csv(MPOX / "forecast_original" / "mpox_case_reports.csv", parse_dates=["epiweek_start"])
    target_week = original.loc[original["type"] == "forecast", "epiweek_start"].max()

    series = {}
    for _name, _group in history_raw.sort_values("date_week_start").groupby("Jurisdiction"):
        series[_name] = _group.set_index("date_week_start")["Cases"].astype(float)

    horizons = {_name: (target_week - _values.index.max()).days // 7 for _name, _values in series.items()}
    assert min(horizons.values()) >= 1
    EXTRA_WEEKS = 3

    recent_start = target_week - pd.Timedelta(weeks=8)
    recent = history_raw[history_raw["date_week_start"] >= recent_start]
    preliminary_share = recent["Week_Type"].str.startswith("Preliminary").mean()
    return EXTRA_WEEKS, horizons, original, series, target_week


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
def _(ROOT, fc, horizons, pd, predictors, series):
    _tables = []
    for _horizon in sorted(set(horizons.values())):
        _names = [_name for _name, _h in horizons.items() if _h == _horizon]
        _tables.append(
            fc.backtest(
                {_name: series[_name].to_numpy() for _name in _names},
                horizon=_horizon,
                predictors=predictors,
                n_windows=20,
                min_context=52,
                weeks={_name: series[_name].index for _name in _names},
            )
        )
    backtest_table = pd.concat(_tables, ignore_index=True)
    backtest_summary = backtest_table.groupby("model").agg(
        mae=("mae", "mean"),
        mase_median=("mase", "median"),
        mase_undefined=("mase", lambda _s: int(_s.isna().sum())),
        wql=("wql", "mean"),
        coverage_80=("coverage_80", "mean"),
        coverage_90=("coverage_90", "mean"),
        windows=("mae", "size"),
    ).sort_values("mae")
    backtest_table.to_csv(ROOT / "data" / "mpox" / "forecast_tabular" / "backtest_windows.csv", index=False)
    backtest_summary
    return


@app.cell
def _(EXTRA_WEEKS, ROOT, fc, predictors, rv, series):
    backtest_month = fc.backtest(
        {_name: _values.to_numpy() for _name, _values in series.items()},
        horizon=1 + EXTRA_WEEKS,
        predictors=predictors,
        n_windows=10,
        min_context=52,
        weeks={_name: _values.index for _name, _values in series.items()},
    )
    backtest_month.to_csv(ROOT / "data" / "mpox" / "forecast_tabular" / "backtest_windows_4w.csv", index=False)
    rv.mpox_summary(backtest_month)
    return


@app.cell
def _(EXTRA_WEEKS, ROOT, fc, horizons, pd, predictors, series, target_week):
    _rows = []
    _path_rows = []
    for _horizon in sorted(set(horizons.values())):
        _names = [_name for _name, _h in horizons.items() if _h == _horizon]
        _arrays = [series[_name].to_numpy() for _name in _names]
        for _model, _predict in predictors.items():
            _outputs = fc.predict_dated(
                _predict, _arrays, _horizon + EXTRA_WEEKS, [series[_name].index.max() for _name in _names]
            )
            for _jurisdiction, _out in zip(_names, _outputs):
                _path_rows.extend(
                    fc.forecast_path_rows(
                        _out, series[_jurisdiction].index.max(), target_week, jurisdiction=_jurisdiction, model=_model
                    )
                )
                _row = {
                    "jurisdiction": _jurisdiction,
                    "model": _model,
                    "last_observed_week": series[_jurisdiction].index.max().strftime("%Y-%m-%d"),
                    "target_week": target_week.strftime("%Y-%m-%d"),
                    "horizon": _horizon,
                    "median": float(_out.median[_horizon - 1]),
                }
                if _out.quantiles is not None:
                    for _label, _lower, _upper in [("80", 0.1, 0.9), ("90", 0.05, 0.95)]:
                        _low, _high = fc.interval(_out, _lower, _upper)
                        _row[f"lower_{_label}"] = float(_low[_horizon - 1])
                        _row[f"upper_{_label}"] = float(_high[_horizon - 1])
                _rows.append(_row)

    forecast_table = pd.DataFrame(_rows)
    forecast_table.to_csv(ROOT / "data" / "mpox" / "forecast_tabular" / "mpox_forecast.csv", index=False)
    forecast_path = pd.DataFrame(_path_rows)
    forecast_path.to_csv(ROOT / "data" / "mpox" / "forecast_tabular" / "mpox_forecast_path.csv", index=False)
    forecast_table
    return forecast_path, forecast_table


@app.cell
def _(forecast_path):
    forecast_path[forecast_path["jurisdiction"] == "California"]
    return


@app.cell
def _(ROOT, forecast_table, original, pd, rv):
    _orig = original[original["type"] == "forecast"]
    _orig_wide = _orig.pivot_table(index="jurisdiction", columns="method", values="predicted", aggfunc="first")
    _orig_wide.columns = [f"original_{_col}" for _col in _orig_wide.columns]
    _model_wide = forecast_table.pivot_table(index="jurisdiction", columns="model", values="median", aggfunc="first")
    _model_wide.columns = [f"{_col}_median" for _col in _model_wide.columns]
    comparison = _model_wide.join(_orig_wide, how="outer").sort_index()
    comparison.to_csv(ROOT / "data" / "mpox" / "forecast_tabular" / "mpox_vs_original.csv")

    matched = comparison.dropna(subset=["timestral_median", "original_VAR_LASSO"])
    _gaps = {
        _model: float((matched[f"{_model}_median"] - matched["original_VAR_LASSO"]).abs().mean())
        for _model in rv.MODELS
    }
    original_only = sorted(set(original["jurisdiction"]) - set(forecast_table["jurisdiction"]))
    mean_abs_gap_to_var_lasso = pd.Series(_gaps, name="mean_abs_gap_to_VAR_LASSO")
    mean_abs_gap_to_var_lasso, original_only
    return (matched,)


@app.cell
def _(matched, plt):
    _models = ["timestral", "timesfm", "tabpfn_ts"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    _top = max(matched["original_VAR_LASSO"].max(), *(matched[f"{_model}_median"].max() for _model in _models))
    for _ax, _model in zip(axes, _models):
        _ax.scatter(matched["original_VAR_LASSO"], matched[f"{_model}_median"], alpha=0.7)
        _ax.plot([0, _top], [0, _top], "k--", linewidth=1)
        _ax.set_title(f"{_model} median vs original VAR_LASSO")
        _ax.set_xlabel("original VAR_LASSO predicted")
        _ax.set_ylabel(f"{_model} median")
    fig.tight_layout()
    fig
    return


if __name__ == "__main__":
    app.run()
