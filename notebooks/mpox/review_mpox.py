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
    # Mpox: review of Timestral, TimesFM 3.0 and TabPFN-TS results

    Reads the saved outputs of `notebooks/mpox/mpox.py`. No model is loaded or run here, so this
    notebook works without downloading the checkpoints.

    The backtest is pooled over the per-horizon rolling-origin windows. The original forecast has
    no observed target week, so the comparison with it measures agreement only, not accuracy.
    MASE is undefined for zero-heavy windows and is reported as a count.

    The model forecasts run one month out: through the target week plus three more weeks. VAR_LASSO
    covers the target week only, so the later weeks have no comparison. The second summary table
    is a separate 4-week backtest (10 windows per jurisdiction) for that range. The jurisdiction plot has three panels: the entire history (top), one full year ending at the last forecast week with earlier years wrapped onto the same axis (middle), and the last 60 days (bottom), all with the forecast path.
    """)
    return


@app.cell
def _(rv):
    windows = rv.mpox_backtest()
    forecast = rv.mpox_forecast()
    comparison = rv.mpox_comparison()
    history = rv.mpox_history()
    windows_4w = rv.mpox_backtest_4w()
    forecast_path = rv.mpox_forecast_path()
    return comparison, forecast, forecast_path, history, windows, windows_4w


@app.cell
def _(rv, windows):
    summary = rv.mpox_summary(windows)
    summary
    return


@app.cell
def _(rv, windows_4w):
    summary_4w = rv.mpox_summary(windows_4w)
    summary_4w
    return


@app.cell
def _(comparison, rv):
    gap = rv.mpox_gap(comparison)
    matched_count = int(comparison.dropna(subset=["timestral_median", "original_VAR_LASSO"]).shape[0])
    gap, matched_count
    return


@app.cell
def _(comparison, plt):
    matched = comparison.dropna(subset=["timestral_median", "original_VAR_LASSO"])
    scatter_models = ["timestral", "timesfm", "tabpfn_ts"]
    scatter_fig, scatter_axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    top = max(matched["original_VAR_LASSO"].max(), *(matched[f"{_model}_median"].max() for _model in scatter_models))
    for _ax, _model in zip(scatter_axes, scatter_models):
        _ax.scatter(matched["original_VAR_LASSO"], matched[f"{_model}_median"], alpha=0.7)
        _ax.plot([0, top], [0, top], "k--", linewidth=1)
        _ax.set_title(f"{_model} median vs original VAR_LASSO")
        _ax.set_xlabel("original VAR_LASSO predicted")
        _ax.set_ylabel(f"{_model} median")
    scatter_fig.tight_layout()
    scatter_fig
    return


@app.cell
def _(forecast, mo):
    jurisdictions = sorted(forecast["jurisdiction"].unique())
    jurisdiction = mo.ui.dropdown(options=jurisdictions, value="California" if "California" in jurisdictions else jurisdictions[0], label="Jurisdiction")
    jurisdiction
    return (jurisdiction,)


@app.cell
def _(comparison, forecast, forecast_path, history, jurisdiction, plt, rv):
    name = jurisdiction.value
    series = history[history["Jurisdiction"] == name].set_index("date_week_start")["Cases"].sort_index()
    rows = forecast[forecast["jurisdiction"] == name]
    original_value = comparison.loc[name, "original_VAR_LASSO"] if name in comparison.index else None

    detail_fig, detail_axes = plt.subplots(3, 1, figsize=(10, 14))
    for _ax, _wrap, _days in [(detail_axes[0], False, None), (detail_axes[1], True, None), (detail_axes[2], False, 60)]:
        rv.plot_mpox_path(
            _ax,
            name,
            series,
            forecast_path[forecast_path["jurisdiction"] == name],
            target_week=rows["target_week"].iloc[0],
            original=original_value,
            weeks=None,
            days=_days,
            wrap_years=_wrap,
        )
    detail_fig.tight_layout()
    detail_fig
    return (name,)


@app.cell
def _(forecast_path, name):
    forecast_path[forecast_path["jurisdiction"] == name].pivot_table(index="week", columns="model", values="median")
    return


if __name__ == "__main__":
    app.run()
