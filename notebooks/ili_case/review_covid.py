import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt

    ROOT = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(ROOT))

    import review as rv

    return mo, plt, rv


@app.cell
def _(mo):
    mo.md("""
    # COVID ILI cases: review of Timestral, TimesFM 3.0 and TabPFN-TS results

    Reads the saved outputs of `notebooks/ili_case/ili_cases.py`. No model is loaded or run here, so this
    notebook works without downloading the checkpoints.

    Results cover a rolling-origin backtest over the history and the 4-week (one month) forecast
    overlaid on the original forecast. The overlay has three panels: the entire observed period (top), one full year ending at the last forecast week with each earlier year wrapped onto the same axis (middle), and the last 60 days with the forecast (bottom). TimesFM 3.0 has no 90% level, so its band is the 80% interval.
    """)
    return


@app.cell
def _():
    DISEASE = "COVID"
    return (DISEASE,)


@app.cell
def _(DISEASE, rv):
    windows = rv.ili_backtest()
    windows = windows[windows["series"] == DISEASE]
    history = rv.ili_history(DISEASE)
    original = rv.ili_original(DISEASE)
    forecast = rv.ili_forecast()
    forecast = forecast[forecast["disease"] == DISEASE]
    return forecast, history, original, windows


@app.cell
def _(rv, windows):
    summary = rv.ili_summary(windows)
    summary
    return


@app.cell
def _(history, plt, rv, windows):
    window_fig, window_axes = plt.subplots(1, 2, figsize=(12, 4))
    rv.plot_ili_windows(window_axes[0], windows, history, metric="mae")
    rv.plot_ili_windows(window_axes[1], windows, history, metric="mase")
    window_fig.tight_layout()
    window_fig
    return


@app.cell
def _(forecast):
    forecast_view = forecast[["model", "week_end", "horizon", "median", "lower_80", "upper_80", "lower_90", "upper_90"]]
    forecast_view
    return


@app.cell
def _(DISEASE, forecast, history, original, plt, rv):
    overlay_fig, overlay_axes = plt.subplots(3, 1, figsize=(10, 14))
    rv.plot_ili_overlay(overlay_axes[0], DISEASE, history, original, forecast, weeks=None)
    rv.plot_ili_overlay(overlay_axes[1], DISEASE, history, original, forecast, wrap_years=True)
    rv.plot_ili_overlay(overlay_axes[2], DISEASE, history, original, forecast, days=60)
    overlay_fig.tight_layout()
    overlay_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Rt (reproduction number)

    Reads the saved outputs of `notebooks/ili_case/rt_ili.py`. The Rt estimate is the renewal-equation value per
    week, with a 90% band. The Rt forecast is the 4-week TabPFN-TS forecast of that series, compared with
    a naive carry-forward. Rt backtest errors are scored against the estimated Rt, not a true Rt, and the
    generation intervals behind the estimate are assumptions (see `notebooks/ili_case/rt_ili.py`).

    The original Rt is a separate daily estimate from the original forecast files. It is drawn for
    visual comparison only, since it uses a different method and a different case series.
    """)
    return


@app.cell
def _(DISEASE, rv):
    rt_windows = rv.rt_backtest(DISEASE)
    rt_estimate = rv.rt_estimate(DISEASE)
    rt_forecast = rv.rt_forecast(DISEASE)
    rt_original = rv.rt_original(DISEASE)
    rt_fit = rv.rt_renewal_fit(DISEASE)
    rt_projection = rv.rt_renewal_projection(DISEASE)
    return (
        rt_estimate,
        rt_fit,
        rt_forecast,
        rt_original,
        rt_projection,
        rt_windows,
    )


@app.cell
def _(rt_windows, rv):
    rt_summary = rv.rt_summary(rt_windows)
    rt_summary
    return


@app.cell
def _(DISEASE, plt, rt_estimate, rt_forecast, rt_original, rv):
    rt_fig, rt_axes = plt.subplots(3, 1, figsize=(10, 14))
    rv.plot_rt_overlay(rt_axes[0], DISEASE, rt_estimate, rt_forecast, rt_original)
    rv.plot_rt_overlay(rt_axes[1], DISEASE, rt_estimate, rt_forecast, rt_original, wrap_years=True)
    rv.plot_rt_overlay(rt_axes[2], DISEASE, rt_estimate, rt_forecast, rt_original, days=60)
    rt_fig.tight_layout()
    rt_fig
    return


@app.cell
def _(rt_forecast):
    rt_forecast_view = rt_forecast[["model", "week", "step", "median", "lower_80", "upper_80", "lower_90", "upper_90"]]
    rt_forecast_view
    return


@app.cell
def _(rt_fit):
    rt_fit
    return


@app.cell
def _(rt_projection):
    rt_projection
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Hospital admissions

    Reads the saved outputs of `notebooks/ili_case/ili_hosp.py` and the observed and original hospital counts in
    `forecast_original/`. The 4-week forecast is for weekly hospital admissions. The backtest has only three
    origins per series because the series is short (18 observed weeks), so these scores are indicative only.
    The most recent weeks are partial-data estimates and may be revised upward. The original file forecasts
    three weeks; the fourth week is model-only.
    """)
    return


@app.cell
def _(DISEASE, rv):
    hosp_history = rv.hosp_history(DISEASE)
    hosp_original = rv.hosp_original(DISEASE)
    hosp_forecast = rv.hosp_forecast(DISEASE)
    hosp_windows = rv.hosp_backtest(DISEASE)
    return hosp_forecast, hosp_history, hosp_original, hosp_windows


@app.cell
def _(hosp_windows, rv):
    hosp_summary = rv.ili_summary(hosp_windows)
    hosp_summary
    return


@app.cell
def _(DISEASE, hosp_forecast, hosp_history, hosp_original, plt, rv):
    hosp_fig, hosp_axes = plt.subplots(3, 1, figsize=(10, 14))
    rv.plot_ili_overlay(hosp_axes[0], DISEASE, hosp_history, hosp_original, hosp_forecast, weeks=None, ylabel="admissions", title=f"{DISEASE} hospital admissions: entire period and forecast")
    rv.plot_ili_overlay(hosp_axes[1], DISEASE, hosp_history, hosp_original, hosp_forecast, wrap_years=True, ylabel="admissions", title=f"{DISEASE} hospital admissions: full year, previous years wrapped")
    rv.plot_ili_overlay(hosp_axes[2], DISEASE, hosp_history, hosp_original, hosp_forecast, days=60, ylabel="admissions", title=f"{DISEASE} hospital admissions: last 60 days and forecast")
    hosp_fig.tight_layout()
    hosp_fig
    return


@app.cell
def _(hosp_forecast):
    hosp_forecast_view = hosp_forecast[["model", "week_end", "horizon", "median", "lower_80", "upper_80", "lower_90", "upper_90"]]
    hosp_forecast_view
    return


if __name__ == "__main__":
    app.run()
