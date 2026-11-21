from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
ILI_DIR = ROOT / "data" / "ili_cases"
MPOX_DIR = ROOT / "data" / "mpox"
ILI_RUN = "notebooks/ili_case/ili_cases.py"
RT_RUN = "notebooks/ili_case/rt_ili.py"
HOSP_RUN = "notebooks/ili_case/ili_hosp.py"
MPOX_RUN = "notebooks/mpox/mpox.py"
MODELS = ["timestral", "timesfm", "tabpfn_ts", "naive", "seasonal_naive"]
FOUNDATION_COLORS = [("timestral", "tab:blue"), ("timesfm", "tab:orange"), ("tabpfn_ts", "tab:green")]


def require(path: Path, run_notebook: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path.relative_to(ROOT)} is missing; run {run_notebook} to generate it")
    return path


def ili_backtest() -> pd.DataFrame:
    return pd.read_csv(require(ILI_DIR / "forecast_tabular" / "backtest_windows.csv", ILI_RUN))


def ili_forecast() -> pd.DataFrame:
    path = require(ILI_DIR / "forecast_tabular" / "ili_cases_forecast.csv", ILI_RUN)
    return pd.read_csv(path, parse_dates=["week_end"])


def ili_history(disease: str) -> pd.Series:
    path = require(ILI_DIR / "input" / f"{disease}.csv", ILI_RUN)
    frame = pd.read_csv(path, parse_dates=["date_week_end"])
    return frame.set_index("date_week_end")["Cases"].astype(float)


def ili_original(disease: str) -> pd.DataFrame:
    path = require(ILI_DIR / "forecast_original" / f"{disease}_case_reports.csv", ILI_RUN)
    frame = pd.read_csv(path, parse_dates=["epiweek_end"])
    return frame[frame["Type"] == "forecast"]


def ili_summary(windows: pd.DataFrame) -> pd.DataFrame:
    return (
        windows.groupby("model")
        .agg(
            mae=("mae", "mean"),
            mase=("mase", "mean"),
            wql=("wql", "mean"),
            coverage_80=("coverage_80", "mean"),
            coverage_90=("coverage_90", "mean"),
            windows=("mae", "size"),
        )
        .sort_values("mae")
    )


def hosp_history(disease: str) -> pd.Series:
    path = require(ILI_DIR / "forecast_original" / f"{disease}_hosp_reports.csv", ILI_RUN)
    frame = pd.read_csv(path, parse_dates=["date"])
    observed = frame[frame["Observations"].notna()]
    return observed.set_index("date")["Observations"].astype(float)


def hosp_original(disease: str) -> pd.DataFrame:
    path = require(ILI_DIR / "forecast_original" / f"{disease}_hosp_reports.csv", ILI_RUN)
    frame = pd.read_csv(path, parse_dates=["date"])
    frame = frame[frame["type"] == "forecast"]
    return frame.rename(columns={"date": "epiweek_end", "median": "Median", "lower_90": "Lower90", "upper_90": "Upper90"})


def hosp_forecast(disease: str) -> pd.DataFrame:
    path = require(ILI_DIR / "forecast_tabular" / "ili_hosp_forecast.csv", HOSP_RUN)
    frame = pd.read_csv(path, parse_dates=["week_end"])
    return frame[frame["disease"] == disease].reset_index(drop=True)


def hosp_backtest(disease: str) -> pd.DataFrame:
    path = require(ILI_DIR / "forecast_tabular" / "hosp_backtest_windows.csv", HOSP_RUN)
    frame = pd.read_csv(path)
    return frame[frame["series"] == disease].reset_index(drop=True)


def rt_estimate(disease: str) -> pd.DataFrame:
    path = require(ILI_DIR / "forecast_tabular" / "rt_estimate.csv", RT_RUN)
    frame = pd.read_csv(path, parse_dates=["week"])
    return frame[frame["disease"] == disease].reset_index(drop=True)


def rt_forecast(disease: str) -> pd.DataFrame:
    path = require(ILI_DIR / "forecast_tabular" / "rt_forecast.csv", RT_RUN)
    frame = pd.read_csv(path, parse_dates=["week"])
    return frame[frame["disease"] == disease].reset_index(drop=True)


def rt_backtest(disease: str) -> pd.DataFrame:
    path = require(ILI_DIR / "forecast_tabular" / "rt_backtest_windows.csv", RT_RUN)
    frame = pd.read_csv(path)
    return frame[frame["series"] == disease].reset_index(drop=True)


def rt_summary(windows: pd.DataFrame) -> pd.DataFrame:
    return (
        windows.groupby("model")
        .agg(
            mae=("mae", "mean"),
            mase=("mase", "mean"),
            wql=("wql", "mean"),
            coverage_80=("coverage_80", "mean"),
            coverage_90=("coverage_90", "mean"),
            windows=("mae", "size"),
        )
        .sort_values("mae")
    )


def rt_renewal_fit(disease: str) -> pd.DataFrame:
    path = require(ILI_DIR / "forecast_tabular" / "rt_renewal_fit.csv", RT_RUN)
    frame = pd.read_csv(path)
    return frame[frame["disease"] == disease].reset_index(drop=True)


def rt_renewal_projection(disease: str) -> pd.DataFrame:
    path = require(ILI_DIR / "forecast_tabular" / "rt_renewal_projection.csv", RT_RUN)
    frame = pd.read_csv(path, parse_dates=["week"])
    return frame[frame["disease"] == disease].reset_index(drop=True)


def rt_original(disease: str) -> pd.DataFrame:
    path = require(ILI_DIR / "forecast_original" / f"{disease}_case_Rt.csv", ILI_RUN)
    frame = pd.read_csv(path, parse_dates=["date"])
    return frame[frame["variable"] == "R"].reset_index(drop=True)


def mpox_backtest() -> pd.DataFrame:
    return pd.read_csv(require(MPOX_DIR / "forecast_tabular" / "backtest_windows.csv", MPOX_RUN))


def mpox_forecast() -> pd.DataFrame:
    path = require(MPOX_DIR / "forecast_tabular" / "mpox_forecast.csv", MPOX_RUN)
    return pd.read_csv(path, parse_dates=["last_observed_week", "target_week"])


def mpox_comparison() -> pd.DataFrame:
    path = require(MPOX_DIR / "forecast_tabular" / "mpox_vs_original.csv", MPOX_RUN)
    return pd.read_csv(path, index_col="jurisdiction")


def mpox_history() -> pd.DataFrame:
    path = require(MPOX_DIR / "input" / "mpox_aggregated_geo.csv", MPOX_RUN)
    return pd.read_csv(
        path,
        usecols=["Jurisdiction", "date_week_start", "Cases"],
        parse_dates=["date_week_start"],
    )


def mpox_summary(windows: pd.DataFrame) -> pd.DataFrame:
    return windows.groupby("model").agg(
        mae=("mae", "mean"),
        mase_median=("mase", "median"),
        mase_undefined=("mase", lambda values: int(values.isna().sum())),
        wql=("wql", "mean"),
        coverage_80=("coverage_80", "mean"),
        coverage_90=("coverage_90", "mean"),
        windows=("mae", "size"),
    ).sort_values("mae")


def mpox_gap(comparison: pd.DataFrame) -> pd.Series:
    matched = comparison.dropna(subset=["timestral_median", "original_VAR_LASSO"])
    gaps = {
        model: float((matched[f"{model}_median"] - matched["original_VAR_LASSO"]).abs().mean())
        for model in MODELS
    }
    return pd.Series(gaps, name="mean_abs_gap_to_VAR_LASSO")


def band(frame: pd.DataFrame) -> tuple[str, pd.Series, pd.Series]:
    if frame["lower_90"].notna().all():
        return "90%", frame["lower_90"], frame["upper_90"]
    return "80%", frame["lower_80"], frame["upper_80"]


def history_window(history: pd.Series, weeks: int | None = None, days: int | None = None) -> pd.Series:
    if days is not None:
        return history[history.index >= history.index.max() - pd.Timedelta(days=days)]
    if weeks is not None:
        return history.iloc[-weeks:]
    return history


def window_label(weeks: int | None = None, days: int | None = None) -> str:
    if days is not None:
        return f"last {days} days and forecast"
    if weeks is not None:
        return f"last {weeks} weeks and forecast"
    return "entire period and forecast"


def year_window(end) -> tuple[pd.Timestamp, pd.Timestamp]:
    end = pd.Timestamp(end)
    return end - pd.DateOffset(years=1), end


def plot_previous_years(ax, history: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> None:
    span = max(1, int((end - history.index.min()).days // 365) + 1)
    shades = ["tab:purple", "tab:brown", "tab:pink", "tab:olive", "tab:cyan", "tab:grey"]
    for k in range(1, span + 1):
        lo, hi = start - pd.DateOffset(years=k), end - pd.DateOffset(years=k)
        rows = history[(history.index >= lo) & (history.index <= hi)]
        if rows.empty:
            continue
        ax.plot(
            rows.index + pd.DateOffset(years=k),
            rows.values,
            color=shades[(k - 1) % len(shades)],
            linewidth=1,
            alpha=0.7,
            label=f"observed {lo:%b %Y} to {hi:%b %Y}",
        )
    ax.set_xlim(start, end + pd.Timedelta(days=7))


def plot_ili_windows(ax, windows: pd.DataFrame, history: pd.Series, metric: str = "mae", models: list[str] = MODELS) -> None:
    for model in models:
        rows = windows[windows["model"] == model]
        ax.plot(history.index[rows["origin"].to_numpy()], rows[metric], marker="o", markersize=3, label=model)
    ax.set_title(f"{metric} per rolling-origin window")
    ax.set_xlabel("first forecast week")
    ax.set_ylabel(metric)
    ax.legend()


def plot_ili_overlay(
    ax,
    disease: str,
    history: pd.Series,
    original: pd.DataFrame,
    forecast: pd.DataFrame,
    weeks: int | None = 52,
    days: int | None = None,
    wrap_years: bool = False,
    ylabel: str = "cases",
    title: str | None = None,
) -> None:
    if wrap_years:
        start, end = year_window(forecast["week_end"].max())
        plot_previous_years(ax, history, start, end)
        recent = history[history.index >= start]
    else:
        recent = history_window(history, weeks=weeks, days=days)
    ax.plot(recent.index, recent.values, color="black", linewidth=2 if wrap_years else 1.5, label="observed")
    ax.plot(original["epiweek_end"], original["Median"], "k--", label="original median")
    ax.fill_between(
        original["epiweek_end"],
        original["Lower90"],
        original["Upper90"],
        color="grey",
        alpha=0.3,
        label="original 90%",
    )
    for model, color in FOUNDATION_COLORS:
        rows = forecast[forecast["model"] == model]
        level, low, high = band(rows)
        ax.plot(rows["week_end"], rows["median"], color=color, marker="o", label=model)
        ax.fill_between(rows["week_end"], low, high, color=color, alpha=0.15, label=f"{model} {level}")
    label = "full year and forecast, previous years wrapped" if wrap_years else window_label(weeks, days)
    ax.set_title(title or f"{disease}: {label}")
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", fontsize=8 if wrap_years else None)


def plot_rt_overlay(
    ax,
    disease: str,
    estimate: pd.DataFrame,
    forecast: pd.DataFrame,
    original: pd.DataFrame,
    weeks: int | None = None,
    days: int | None = None,
    wrap_years: bool = False,
) -> None:
    rt = estimate.set_index("week")
    if wrap_years:
        start, end = year_window(forecast["week"].max())
        plot_previous_years(ax, rt["rt"], start, end)
        rt = rt[rt.index >= start]
    else:
        rt = rt.loc[history_window(rt["rt"], weeks=weeks, days=days).index]
    ax.fill_between(rt.index, rt["rt_lower_90"], rt["rt_upper_90"], color="black", alpha=0.1, label="Rt estimate 90%")
    ax.plot(rt.index, rt["rt"], color="black", marker="o", markersize=3, linewidth=2 if wrap_years else 1.5, label="Rt estimate (renewal)")
    original = original[original["date"] >= rt.index.min()]
    estimated = original[original["type"] != "forecast"]
    projected = original[original["type"] == "forecast"]
    ax.plot(estimated["date"], estimated["median"], "k--", linewidth=1, label="original Rt (daily)")
    ax.plot(projected["date"], projected["median"], "k:", linewidth=1, label="original Rt forecast (daily)")
    ax.fill_between(original["date"], original["lower_90"], original["upper_90"], color="grey", alpha=0.2, label="original Rt 90%")
    for model, color in [("tabpfn_ts", "tab:green"), ("naive", "tab:grey")]:
        rows = forecast[forecast["model"] == model]
        ax.plot(rows["week"], rows["median"], color=color, marker="o", linestyle="-" if model == "tabpfn_ts" else ":", label=f"{model} Rt")
        if model == "tabpfn_ts":
            level, low, high = band(rows)
            ax.fill_between(rows["week"], low, high, color=color, alpha=0.15, label=f"{model} Rt {level}")
    ax.axhline(1.0, color="red", linewidth=0.8, linestyle=":", label="Rt = 1")
    label = "full year and forecast, previous years wrapped" if wrap_years else window_label(weeks, days)
    ax.set_title(f"{disease} Rt: {label}")
    ax.set_ylabel("Rt")
    ax.legend(loc="upper left", fontsize=8)


MPOX_REGIONAL_RUN = "notebooks/mpox/mpox_regional.py"
MPOX_REGIONS = {
    "San Diego": "SanDiego",
    "Los Angeles": "LosAngelesCounty",
    "San Francisco": "SanFranciscoCounty",
    "New York City": "NewYorkCity",
    "California": "California",
}
ORIGINAL_METHODS = ["NAIVE", "AR_LASSO", "VAR_LASSO"]


def mpox_case_reports() -> pd.DataFrame:
    path = require(MPOX_DIR / "forecast_original" / "mpox_case_reports.csv", MPOX_REGIONAL_RUN)
    return pd.read_csv(path, parse_dates=["epiweek_start"])


def mpox_regional_history() -> pd.DataFrame:
    reports = mpox_case_reports()
    rows = reports[(reports["type"] == "observation") & reports["jurisdiction"].isin(MPOX_REGIONS.values())]
    return rows[["jurisdiction", "epiweek_start", "cases"]].sort_values(["jurisdiction", "epiweek_start"])


def mpox_original_forecast() -> pd.DataFrame:
    reports = mpox_case_reports()
    return reports[reports["type"] == "forecast"]


def mpox_regional_backtest() -> pd.DataFrame:
    path = require(MPOX_DIR / "forecast_tabular" / "regional_backtest_windows.csv", MPOX_REGIONAL_RUN)
    return pd.read_csv(path)


def mpox_regional_forecast() -> pd.DataFrame:
    path = require(MPOX_DIR / "forecast_tabular" / "regional_forecast.csv", MPOX_REGIONAL_RUN)
    return pd.read_csv(path, parse_dates=["last_context_week", "target_week"])


def mpox_regional_summary(windows: pd.DataFrame) -> pd.DataFrame:
    return (
        windows.groupby(["series", "horizon", "model"])
        .agg(
            mae=("mae", "mean"),
            mase=("mase", "mean"),
            mase_undefined=("mase", lambda values: int(values.isna().sum())),
            wql=("wql", "mean"),
            wql_undefined=("wql", lambda values: int(values.isna().sum())),
            coverage_80=("coverage_80", "mean"),
            coverage_90=("coverage_90", "mean"),
            windows=("mae", "size"),
        )
        .loc[list(MPOX_REGIONS)]
    )


def mpox_regional_pivot(windows: pd.DataFrame, metric: str = "mae") -> pd.DataFrame:
    table = windows.pivot_table(index=["series", "horizon"], columns="model", values=metric, aggfunc="mean")
    return table.loc[list(MPOX_REGIONS), MODELS]


def mpox_regional_target_errors(forecast: pd.DataFrame) -> pd.DataFrame:
    scored = forecast.dropna(subset=["observed"])
    table = scored.pivot_table(index="region", columns="model", values="median", aggfunc="first")[MODELS]
    first = scored.groupby("region").first()
    for method in ORIGINAL_METHODS:
        table[f"original_{method}"] = first[f"original_{method}"]
    errors = table.sub(first["observed"], axis=0).abs()
    errors.insert(0, "observed", first["observed"])
    return errors


def plot_mpox_region(ax, region: str, history: pd.Series, rows: pd.DataFrame, weeks: int = 52) -> None:
    recent = history.iloc[-weeks:]
    ax.plot(recent.index, recent.values, color="black", marker="o", markersize=3, label="observed")
    target = pd.Timestamp(rows["target_week"].iloc[0])
    first = rows.iloc[0]
    for model, color in FOUNDATION_COLORS:
        row = rows[rows["model"] == model].iloc[0]
        level = "90" if pd.notna(row["lower_90"]) else "80"
        ax.errorbar(
            [target],
            [row["median"]],
            yerr=[[row["median"] - row[f"lower_{level}"]], [row[f"upper_{level}"] - row["median"]]],
            fmt="o",
            color=color,
            capsize=4,
            label=f"{model} ({level}%)",
        )
    if pd.notna(first["original_VAR_LASSO"]):
        ax.plot([target], [first["original_VAR_LASSO"]], "kx", markersize=10, label="original VAR_LASSO")
    if pd.notna(first["observed"]):
        ax.plot([target], [first["observed"]], "r*", markersize=12, label="observed target week (held out)")
    ax.set_title(f"{region} (forecast {int(first['horizon'])} week(s) ahead)")
    ax.set_ylabel("cases per week")
    ax.legend(loc="upper left", fontsize=8)


def mpox_backtest_4w() -> pd.DataFrame:
    return pd.read_csv(require(MPOX_DIR / "forecast_tabular" / "backtest_windows_4w.csv", MPOX_RUN))


def mpox_forecast_path() -> pd.DataFrame:
    path = require(MPOX_DIR / "forecast_tabular" / "mpox_forecast_path.csv", MPOX_RUN)
    return pd.read_csv(path, parse_dates=["week"])


def mpox_regional_forecast_path() -> pd.DataFrame:
    path = require(MPOX_DIR / "forecast_tabular" / "regional_forecast_path.csv", MPOX_REGIONAL_RUN)
    return pd.read_csv(path, parse_dates=["week"])


def plot_mpox_path(
    ax,
    name: str,
    history: pd.Series,
    path: pd.DataFrame,
    target_week,
    original=None,
    observed=None,
    weeks: int | None = 52,
    days: int | None = None,
    wrap_years: bool = False,
) -> None:
    target = pd.Timestamp(target_week)
    observed_history = history[history.index < target]
    if wrap_years:
        last = pd.to_datetime(path["week"]).max() if len(path) else target
        start, end = year_window(max(last, target))
        plot_previous_years(ax, observed_history, start, end)
        recent = observed_history[observed_history.index >= start]
    else:
        recent = history_window(observed_history, weeks=weeks, days=days)
    ax.plot(recent.index, recent.values, color="black", marker="o", markersize=3, label="observed")
    for model, color in FOUNDATION_COLORS:
        rows = path[path["model"] == model].sort_values("week")
        if rows.empty:
            continue
        weeks_ = pd.to_datetime(rows["week"])
        level, low, high = band(rows)
        ax.plot(weeks_, rows["median"], color=color, marker="o", markersize=3, label=model)
        ax.fill_between(weeks_, low, high, color=color, alpha=0.15, label=f"{model} {level}")
    ax.axvline(target, color="grey", linestyle=":", linewidth=1)
    if original is not None and pd.notna(original):
        ax.plot([target], [original], "kx", markersize=10, label="original VAR_LASSO (target week only)")
    if observed is not None and pd.notna(observed):
        ax.plot([target], [observed], "r*", markersize=12, label="observed target week (held out)")
    label = "full year and forecast, previous years wrapped" if wrap_years else window_label(weeks, days)
    ax.set_title(f"{name}: {label} (path to 3 weeks past the target week)")
    ax.set_ylabel("cases per week")
    ax.legend(loc="upper left", fontsize=8)
