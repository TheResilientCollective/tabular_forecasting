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
    # Mpox, regional: review of Timestral, TimesFM 3.0 and TabPFN-TS results

    Reads the saved outputs of `notebooks/mpox/mpox_regional.py` for San Diego, Los Angeles, San
    Francisco, New York City and California. No model is loaded or run here.

    The region plot has three panels: the entire history (top), one full year ending at the last
    forecast week with earlier years wrapped onto the same axis (middle), and the last 60 days
    (bottom), all with the forecast path.

    Caveats:

    - The backtest windows are per region and per horizon (1 and 4 weeks), with 20 windows each.
    - The target-week comparison uses a single week per region. Only San Diego and San Francisco
      have an observed target week, so they are the only regions that can be scored against it,
      and that is one week each. This is not a general accuracy claim.
    - Los Angeles data stops 12 weeks before the target week, so its forecast is 12 weeks ahead.
    - The model forecasts run one month out: through the target week plus three more weeks. The
      original forecast covers the target week only, so the later weeks have no comparison; the
      4-week backtest rows are the evidence for that range.
    - MASE is undefined when a context window has no variation; the count is reported.
    """)
    return


@app.cell
def _(rv):
    windows = rv.mpox_regional_backtest()
    forecast = rv.mpox_regional_forecast()
    history = rv.mpox_regional_history()
    forecast_path = rv.mpox_regional_forecast_path()
    return forecast, forecast_path, history, windows


@app.cell
def _(rv, windows):
    summary = rv.mpox_regional_summary(windows)
    summary
    return


@app.cell
def _(rv, windows):
    rv.mpox_regional_pivot(windows, "mase")
    return


@app.cell
def _(rv, windows):
    rv.mpox_regional_pivot(windows, "mae")
    return


@app.cell
def _(forecast, rv):
    rv.mpox_regional_target_errors(forecast)
    return


@app.cell
def _(forecast, mo):
    regions = list(forecast["region"].unique())
    region = mo.ui.dropdown(options=regions, value="California" if "California" in regions else regions[0], label="Region")
    region
    return (region,)


@app.cell
def _(forecast, forecast_path, history, plt, region, rv):
    _name = region.value
    _rows = forecast[forecast["region"] == _name]
    _values = history[history["jurisdiction"] == rv.MPOX_REGIONS[_name]].set_index("epiweek_start")["cases"]
    _fig, _axes = plt.subplots(3, 1, figsize=(10, 14))
    for _ax, _wrap, _days in [(_axes[0], False, None), (_axes[1], True, None), (_axes[2], False, 60)]:
        rv.plot_mpox_path(
            _ax,
            _name,
            _values,
            forecast_path[forecast_path["region"] == _name],
            target_week=_rows["target_week"].iloc[0],
            original=_rows["original_VAR_LASSO"].iloc[0],
            observed=_rows["observed"].iloc[0],
            days=_days,
            wrap_years=_wrap,
        )
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(forecast_path, region):
    forecast_path[forecast_path["region"] == region.value].pivot_table(index="week", columns="model", values="median")
    return


if __name__ == "__main__":
    app.run()
