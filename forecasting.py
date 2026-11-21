from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TIMESTRAL_CHECKPOINT = "mistralai/Timestral-5M"
TIMESFM_CHECKPOINT = "google/timesfm-3.0-pytorch"
SEASON_LENGTH = 52
PATH_EXTRA_WEEKS = 3
TABPFN_TS_LEVELS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
TABPFN_TS_DEFAULT_END = pd.Timestamp("2026-01-04")
ENV_FILE = Path(__file__).resolve().parent / ".env"


@dataclass
class Forecast:
    median: np.ndarray
    quantiles: np.ndarray | None = None
    levels: tuple[float, ...] = ()


Predictor = Callable[[list[np.ndarray], int], list[Forecast]]


def load_timestral(checkpoint: str = TIMESTRAL_CHECKPOINT, device: str = "cpu"):
    from timestral import TimestralForecaster

    return TimestralForecaster.from_pretrained(checkpoint, device=device)


def load_timesfm(device: str = "cpu"):
    from timesfm3 import ModelConfig, TimesFM3Evaluator

    config = ModelConfig(device=device)
    return TimesFM3Evaluator(config), tuple(config.quantiles)


def timestral_predictor(model) -> Predictor:
    def predict(histories: list[np.ndarray], horizon: int) -> list[Forecast]:
        levels = tuple(model.quantile_levels)
        outputs = []
        for history in histories:
            result = model.forecast(np.asarray(history, dtype=np.float32), prediction_length=horizon)
            quantiles = np.clip(result.quantiles[0, 0].cpu().numpy(), 0, None)
            outputs.append(Forecast(median=quantiles[:, model.median_index], quantiles=quantiles, levels=levels))
        return outputs

    return predict


def timesfm_predictor(evaluator, levels: tuple[float, ...]) -> Predictor:
    def predict(histories: list[np.ndarray], horizon: int) -> list[Forecast]:
        arrays = [np.asarray(history, dtype=np.float32) for history in histories]
        outputs = list(evaluator.predict_batch(arrays, horizon=horizon, return_quantiles=True))
        return [
            Forecast(
                median=np.clip(np.asarray(out.forecast), 0, None),
                quantiles=np.clip(np.asarray(out.quantiles), 0, None),
                levels=levels,
            )
            for out in outputs
        ]

    return predict


def load_env_token(name: str = "TABPFN_TOKEN") -> bool:
    if os.environ.get(name):
        return True
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            key, _, value = line.strip().removeprefix("export ").partition("=")
            if key.strip() == name and value.strip():
                os.environ[name] = value.strip().strip("\"'")
                return True
    return False


def load_tabpfn_ts():
    load_env_token()
    from tabpfn_time_series import TabPFNMode, TabPFNTSPipeline

    return TabPFNTSPipeline(tabpfn_mode=TabPFNMode.LOCAL)


def tabpfn_ts_predictor(pipeline, levels: tuple[float, ...] = TABPFN_TS_LEVELS, clip: bool = True) -> Predictor:
    median_index = levels.index(0.5)

    def predict(histories: list[np.ndarray], horizon: int, ends: list[pd.Timestamp] | None = None) -> list[Forecast]:
        frames = []
        for item, history in enumerate(histories):
            end = pd.Timestamp(ends[item]) if ends is not None else TABPFN_TS_DEFAULT_END
            frames.append(
                pd.DataFrame(
                    {
                        "item_id": item,
                        "timestamp": pd.date_range(end=end, periods=len(history), freq="7D"),
                        "target": np.asarray(history, dtype=float),
                    }
                )
            )
        result = pipeline.predict_df(pd.concat(frames, ignore_index=True), prediction_length=horizon, quantiles=list(levels))
        outputs = []
        for item in range(len(histories)):
            quantiles = result.xs(item, level="item_id")[list(levels)].to_numpy(dtype=float)
            if clip:
                quantiles = np.clip(quantiles, 0, None)
            outputs.append(Forecast(median=quantiles[:, median_index], quantiles=quantiles, levels=levels))
        return outputs

    predict.uses_dates = True
    return predict


def predict_dated(predict: Predictor, histories: list[np.ndarray], horizon: int, ends: list[pd.Timestamp]) -> list[Forecast]:
    if getattr(predict, "uses_dates", False):
        return predict(histories, horizon, ends=ends)
    return predict(histories, horizon)


def naive_predictor(histories: list[np.ndarray], horizon: int) -> list[Forecast]:
    return [Forecast(median=np.repeat(float(history[-1]), horizon)) for history in histories]


def seasonal_naive_predictor(histories: list[np.ndarray], horizon: int) -> list[Forecast]:
    outputs = []
    for history in histories:
        if len(history) < SEASON_LENGTH:
            outputs.append(Forecast(median=np.repeat(float(history[-1]), horizon)))
            continue
        last_season = history[-SEASON_LENGTH:]
        outputs.append(Forecast(median=last_season[np.arange(horizon) % SEASON_LENGTH]))
    return outputs


def pinball(actual: np.ndarray, predicted: np.ndarray, level: float) -> np.ndarray:
    diff = actual - predicted
    return np.maximum(level * diff, (level - 1) * diff)


def weighted_quantile_loss(actual: np.ndarray, forecast: Forecast) -> float:
    denominator = np.abs(actual).sum()
    if forecast.quantiles is None or denominator == 0:
        return np.nan
    losses = [
        pinball(actual, forecast.quantiles[:, i], level).sum()
        for i, level in enumerate(forecast.levels)
    ]
    return float(sum(losses) / (len(forecast.levels) * denominator))


def interval(forecast: Forecast, lower: float, upper: float) -> tuple[np.ndarray, np.ndarray]:
    horizon = len(forecast.median)
    levels = np.asarray(forecast.levels)
    low_idx = np.flatnonzero(np.isclose(levels, lower))
    high_idx = np.flatnonzero(np.isclose(levels, upper))
    if forecast.quantiles is None or len(low_idx) == 0 or len(high_idx) == 0:
        return np.full(horizon, np.nan), np.full(horizon, np.nan)
    return forecast.quantiles[:, low_idx[0]], forecast.quantiles[:, high_idx[0]]


def interval_coverage(actual: np.ndarray, forecast: Forecast, lower: float, upper: float) -> float:
    low, high = interval(forecast, lower, upper)
    if np.isnan(low).all():
        return np.nan
    return float(np.mean((actual >= low) & (actual <= high)))


def path_table(forecast: Forecast, last_week: pd.Timestamp) -> pd.DataFrame:
    steps = np.arange(1, len(forecast.median) + 1)
    frame = pd.DataFrame(
        {
            "week": [last_week + pd.Timedelta(weeks=int(step)) for step in steps],
            "step": steps,
            "median": forecast.median,
        }
    )
    for label, lower, upper in [("80", 0.1, 0.9), ("90", 0.05, 0.95)]:
        low, high = interval(forecast, lower, upper)
        frame[f"lower_{label}"] = low
        frame[f"upper_{label}"] = high
    return frame


def mase_scale(history: np.ndarray) -> float:
    return float(np.mean(np.abs(np.diff(history)))) if len(history) > 1 else 0.0


def rolling_origins(length: int, horizon: int, n_windows: int, min_context: int) -> list[int]:
    origins = [length - horizon - k * horizon for k in range(n_windows)]
    return [origin for origin in origins if origin >= min_context]


def backtest(
    series: dict[str, np.ndarray],
    horizon: int,
    predictors: dict[str, Predictor],
    n_windows: int = 4,
    min_context: int = 52,
    weeks: dict[str, pd.DatetimeIndex] | None = None,
) -> pd.DataFrame:
    series = {name: np.asarray(values, dtype=float) for name, values in series.items()}
    windows = [
        (name, origin)
        for name, values in series.items()
        for origin in rolling_origins(len(values), horizon, n_windows, min_context)
    ]
    batched = {}
    for model_name, predict in predictors.items():
        if windows and getattr(predict, "uses_dates", False):
            ends = [weeks[name][origin - 1] for name, origin in windows] if weeks is not None else None
            outputs = predict([series[name][:origin] for name, origin in windows], horizon, ends=ends)
            batched[model_name] = dict(zip(windows, outputs))

    rows = []
    for name, values in series.items():
        origins = rolling_origins(len(values), horizon, n_windows, min_context)
        for origin in origins:
            history = values[:origin]
            actual = values[origin : origin + horizon]
            scale = mase_scale(history)
            for model_name, predict in predictors.items():
                if model_name in batched:
                    forecast = batched[model_name][(name, origin)]
                else:
                    forecast = predict([history], horizon)[0]
                mae = float(np.mean(np.abs(actual - forecast.median)))
                rows.append(
                    {
                        "series": name,
                        "origin": origin,
                        "model": model_name,
                        "mae": mae,
                        "mase": mae / scale if scale > 0 else np.nan,
                        "wql": weighted_quantile_loss(actual, forecast),
                        "coverage_80": interval_coverage(actual, forecast, 0.1, 0.9),
                        "coverage_90": interval_coverage(actual, forecast, 0.05, 0.95),
                    }
                )
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby("model")
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


def forecast_path_rows(forecast: Forecast, last_week: pd.Timestamp, target_week: pd.Timestamp, **keys) -> list[dict]:
    rows = []
    for step in range(len(forecast.median)):
        week = last_week + pd.Timedelta(weeks=step + 1)
        row = {
            **keys,
            "week": week.strftime("%Y-%m-%d"),
            "step": step + 1,
            "weeks_from_target": (week - target_week).days // 7,
            "median": float(forecast.median[step]),
        }
        for label, lower, upper in [("80", 0.1, 0.9), ("90", 0.05, 0.95)]:
            low, high = interval(forecast, lower, upper)
            row[f"lower_{label}"] = float(low[step])
            row[f"upper_{label}"] = float(high[step])
        rows.append(row)
    return rows
