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
    import renewal as rn
    import review as rv

    return fc, mo, np, pd, plt, rn, rv


@app.cell
def _(mo):
    mo.md("""
    # Rt for COVID, flu and RSV: renewal equation, TabPFN-TS and TabPFN Bayesian optimisation

    Rt is not observed, so it cannot be predicted from case counts without a transmission model.
    This notebook combines a renewal equation with the TabPFN tools in two ways:

    1. **Renewal estimate, then TabPFN-TS forecast.** Weekly cases are spread evenly over the days
       of the week, and Rt per week is the Cori-style renewal estimate (cases divided by the
       infection pressure from earlier cases, with a weak gamma prior). TabPFN-TS then forecasts
       that weekly Rt series four weeks ahead.
    2. **Renewal fit with TabPFN Bayesian optimisation.** A renewal model with two parameters, the
       Rt level at the middle of the last 8 weeks and its weekly log-trend, is fitted to the
       observed cases by minimising the root-mean-square error on log cases. The search uses the
       TabPFN surrogate with expected improvement from `tabpfn_extensions.bayesian_optimization`
       (8 starting points plus 40 proposals), and is checked against a plain grid search.

    Caveats:

    - The generation intervals in `renewal.GENERATION_INTERVALS` (mean and sd in days) are assumed
      values and should be replaced with ones you trust. Rt depends directly on them.
    - Cases are treated as infections: there is no reporting delay, no under-reporting and no
      right-truncation correction, and recent weeks may be revised.
    - The Rt intervals from the renewal estimate assume Poisson noise and are too narrow.
    - The Rt backtest scores forecasts against the estimated Rt, not against a true Rt.
    - The fitted model has two parameters and a cheap objective, so a grid search solves it as
      well. Bayesian optimisation is shown here as a working pattern for costlier models.
    """)
    return


@app.cell
def _(pd, rn, rv):
    DISEASES = ["COVID", "FLU", "RSV"]
    OUT = rv.ILI_DIR / "forecast_tabular"
    HORIZON = 4
    FIT_WEEKS = 8

    history = {_disease: rv.ili_history(_disease) for _disease in DISEASES}
    intervals = {_disease: rn.generation_interval(*rn.GENERATION_INTERVALS[_disease]) for _disease in DISEASES}
    estimates = {_disease: rn.estimate_rt(history[_disease], intervals[_disease]) for _disease in DISEASES}
    rt_estimate = pd.concat(
        [_frame.assign(disease=_disease) for _disease, _frame in estimates.items()], ignore_index=True
    )
    rt_estimate.to_csv(OUT / "rt_estimate.csv", index=False)
    rt_estimate.groupby("disease").tail(4)
    return DISEASES, FIT_WEEKS, HORIZON, OUT, estimates, history, intervals


@app.cell
def _(fc):
    predictors = {
        "tabpfn_ts": fc.tabpfn_ts_predictor(fc.load_tabpfn_ts()),
        "naive": fc.naive_predictor,
    }
    return (predictors,)


@app.cell
def _(DISEASES, HORIZON, OUT, estimates, fc, pd, predictors):
    rt_backtest = fc.backtest(
        {_disease: estimates[_disease]["rt"].to_numpy() for _disease in DISEASES},
        horizon=HORIZON,
        predictors=predictors,
        n_windows=20,
        min_context=52,
        weeks={_disease: pd.DatetimeIndex(estimates[_disease]["week"]) for _disease in DISEASES},
    )
    rt_backtest.to_csv(OUT / "rt_backtest_windows.csv", index=False)
    rt_backtest.groupby(["series", "model"]).agg(
        mae=("mae", "mean"),
        mase=("mase", "mean"),
        coverage_80=("coverage_80", "mean"),
        coverage_90=("coverage_90", "mean"),
        windows=("mae", "size"),
    )
    return


@app.cell
def _(DISEASES, HORIZON, OUT, estimates, fc, pd, predictors):
    _frames = []
    for _model, _predict in predictors.items():
        _outputs = fc.predict_dated(
            _predict,
            [estimates[_disease]["rt"].to_numpy() for _disease in DISEASES],
            HORIZON,
            [estimates[_disease]["week"].max() for _disease in DISEASES],
        )
        for _disease, _out in zip(DISEASES, _outputs):
            _table = fc.path_table(_out, estimates[_disease]["week"].max())
            _table.insert(0, "model", _model)
            _table.insert(0, "disease", _disease)
            _frames.append(_table)
    rt_forecast = pd.concat(_frames, ignore_index=True)
    rt_forecast.to_csv(OUT / "rt_forecast.csv", index=False)
    rt_forecast[rt_forecast["model"] == "tabpfn_ts"]
    return (rt_forecast,)


@app.cell
def _(DISEASES, estimates, plt, rt_forecast):
    fig, axes = plt.subplots(len(DISEASES), 1, figsize=(10, 3.4 * len(DISEASES)))
    for _ax, _disease in zip(axes, DISEASES):
        _recent = estimates[_disease].tail(52)
        _ax.plot(_recent["week"], _recent["rt"], color="black", label="renewal estimate")
        _ax.fill_between(_recent["week"], _recent["rt_lower_90"], _recent["rt_upper_90"], color="grey", alpha=0.3)
        _rows = rt_forecast[(rt_forecast["disease"] == _disease) & (rt_forecast["model"] == "tabpfn_ts")]
        _ax.plot(_rows["week"], _rows["median"], color="tab:green", marker="o", label="TabPFN-TS forecast")
        _ax.fill_between(_rows["week"], _rows["lower_90"], _rows["upper_90"], color="tab:green", alpha=0.15, label="TabPFN-TS 90%")
        _ax.axhline(1.0, color="grey", linestyle=":", linewidth=1)
        _ax.set_title(f"{_disease}: weekly Rt")
        _ax.set_ylabel("Rt")
        _ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig
    return


@app.cell
def _(DISEASES, FIT_WEEKS, OUT, estimates, fc, history, intervals, pd, rn):
    fc.load_env_token()
    _rows = []
    for _disease in DISEASES:
        _objective = rn.renewal_objective(history[_disease].to_numpy(), intervals[_disease], FIT_WEEKS)
        _trace = rn.bo_maximise(_objective, n_init=8, n_iter=40, seed=0)
        _best = _trace.loc[_trace["objective"].idxmax()]
        _level, _trend = rn.scale_params(_best[["x0", "x1"]].to_numpy())
        _grid_x, _grid_value = rn.grid_maximise(_objective)
        _grid_level, _grid_trend = rn.scale_params(_grid_x)
        _rows.append(
            {
                "disease": _disease,
                "bo_level": _level,
                "bo_trend": _trend,
                "bo_rt_last_week": float(rn.rt_path(_level, _trend, FIT_WEEKS)[-1]),
                "bo_objective": float(_best["objective"]),
                "best_initial_objective": float(_trace.loc[_trace["phase"] == "init", "objective"].max()),
                "grid_level": _grid_level,
                "grid_trend": _grid_trend,
                "grid_rt_last_week": float(rn.rt_path(_grid_level, _grid_trend, FIT_WEEKS)[-1]),
                "grid_objective": _grid_value,
                "renewal_estimate_last_week": float(estimates[_disease]["rt"].iloc[-1]),
            }
        )
    renewal_fit = pd.DataFrame(_rows)
    renewal_fit.to_csv(OUT / "rt_renewal_fit.csv", index=False)
    renewal_fit
    return (renewal_fit,)


@app.cell
def _(FIT_WEEKS, HORIZON, OUT, history, intervals, np, pd, renewal_fit, rn):
    _frames = []
    for _fit in renewal_fit.itertuples():
        _weekly = history[_fit.disease]
        _daily = rn.daily_incidence(_weekly.to_numpy())
        _trend_rt = rn.rt_path(_fit.bo_level, _fit.bo_trend, FIT_WEEKS, HORIZON)[FIT_WEEKS:]
        _held_rt = np.repeat(_fit.bo_rt_last_week, HORIZON)
        _frames.append(
            pd.DataFrame(
                {
                    "disease": _fit.disease,
                    "week": [_weekly.index.max() + pd.Timedelta(weeks=_step) for _step in range(1, HORIZON + 1)],
                    "rt_trend_continued": _trend_rt,
                    "cases_trend_continued": rn.simulate_weekly(_daily, intervals[_fit.disease], _trend_rt),
                    "rt_held": _held_rt,
                    "cases_rt_held": rn.simulate_weekly(_daily, intervals[_fit.disease], _held_rt),
                }
            )
        )
    renewal_projection = pd.concat(_frames, ignore_index=True)
    renewal_projection.to_csv(OUT / "rt_renewal_projection.csv", index=False)
    renewal_projection
    return (renewal_projection,)


@app.cell
def _(FIT_WEEKS, history, intervals, plt, renewal_fit, renewal_projection, rn):
    fit_fig, fit_axes = plt.subplots(len(renewal_fit), 1, figsize=(10, 3.4 * len(renewal_fit)))
    for _ax, _fit in zip(fit_axes, renewal_fit.itertuples()):
        _weekly = history[_fit.disease]
        _recent = _weekly.iloc[-26:]
        _ax.plot(_recent.index, _recent.values, color="black", marker="o", markersize=3, label="observed")
        _fitted = rn.simulate_weekly(
            rn.daily_incidence(_weekly.to_numpy()[:-FIT_WEEKS]),
            intervals[_fit.disease],
            rn.rt_path(_fit.bo_level, _fit.bo_trend, FIT_WEEKS),
        )
        _ax.plot(_weekly.index[-FIT_WEEKS:], _fitted, color="tab:purple", label="renewal fit (BO)")
        _rows = renewal_projection[renewal_projection["disease"] == _fit.disease]
        _ax.plot(_rows["week"], _rows["cases_trend_continued"], color="tab:purple", linestyle="--", label="projection, trend continued")
        _ax.plot(_rows["week"], _rows["cases_rt_held"], color="tab:purple", linestyle=":", label="projection, Rt held")
        _ax.set_title(f"{_fit.disease}: renewal fit, Rt last week {_fit.bo_rt_last_week:.2f}")
        _ax.set_ylabel("cases")
        _ax.legend(loc="upper left", fontsize=8)
    fit_fig.tight_layout()
    fit_fig
    return


if __name__ == "__main__":
    app.run()
