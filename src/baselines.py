"""
Baseline forecast models.

These must be computed BEFORE any ML model to establish benchmarks.
Both baselines use only information available before the forecast date
(no leakage).

1. Seasonal Naive: forecast = sales from the same weekday, one week ago.
   For h-step ahead: forecast_t+h = actual_t+h-7
   (i.e., repeat the last observed week)

2. Moving Average (28-day): forecast = mean of the last 28 days.
   For h-step ahead: forecast_t+h = mean(actual_{t-27}...actual_t)
   (i.e., the same flat forecast for all h steps)
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple


def seasonal_naive_forecast(
    series: np.ndarray,
    horizon: int = 28,
    season: int = 7,
) -> np.ndarray:
    """
    Seasonal naive: repeat the last `season`-length block.

    For daily data with season=7, horizon=28:
    forecast[0..6]   = series[-7..]    (last week)
    forecast[7..13]  = series[-7..]    (same week repeated)
    forecast[14..20] = series[-7..]    (same week repeated)
    forecast[21..27] = series[-7..]    (same week repeated)

    Parameters
    ----------
    series : np.ndarray
        Historical sales (training period), shape (T,).
    horizon : int
        Number of days to forecast.
    season : int
        Seasonal period (7 = weekly).

    Returns
    -------
    np.ndarray
        Forecasts of shape (horizon,).
    """
    last_season = series[-season:]
    # Tile the last season to cover the horizon
    n_repeats = (horizon // season) + 1
    tiled = np.tile(last_season, n_repeats)
    return tiled[:horizon].astype(np.float64)


def moving_average_forecast(
    series: np.ndarray,
    horizon: int = 28,
    window: int = 28,
) -> np.ndarray:
    """
    Moving average: flat forecast = mean of the last `window` days.

    Parameters
    ----------
    series : np.ndarray
        Historical sales (training period), shape (T,).
    horizon : int
        Number of days to forecast.
    window : int
        Number of trailing days to average.

    Returns
    -------
    np.ndarray
        Flat forecast of shape (horizon,), all values identical.
    """
    if len(series) < window:
        avg = np.mean(series)
    else:
        avg = np.mean(series[-window:])
    return np.full(horizon, avg, dtype=np.float64)


def compute_baselines(
    silver: pd.DataFrame,
    test_days: int = 28,
    season: int = 7,
    ma_window: int = 28,
) -> pd.DataFrame:
    """
    Compute seasonal naive and moving average baselines for all series.

    Uses time-based split: last `test_days` days = test, rest = train.
    No information from the test period is used.

    Parameters
    ----------
    silver : pd.DataFrame
        Silver table with columns: id, date, sales (sorted by id, date).
    test_days : int
        Number of final days to hold out as test.
    season : int
        Seasonal period for naive baseline.
    ma_window : int
        Window for moving average baseline.

    Returns
    -------
    pd.DataFrame
        Test-period dataframe with columns:
        id, date, actual, snaive_forecast, ma_forecast, cat_id, store_id, state_id
    """
    # Determine cutoff
    all_dates = sorted(silver["date"].unique())
    cutoff_date = all_dates[-(test_days)]
    train_dates = [d for d in all_dates if d < cutoff_date]
    test_dates = [d for d in all_dates if d >= cutoff_date]

    print(f"Cutoff date: {cutoff_date}")
    print(f"Train: {len(train_dates)} days ({train_dates[0]} to {train_dates[-1]})")
    print(f"Test:  {len(test_dates)} days ({test_dates[0]} to {test_dates[-1]})")

    assert len(test_dates) == test_days, \
        f"Expected {test_days} test days, got {len(test_dates)}"

    results = []
    series_ids = silver["id"].unique()

    for sid in series_ids:
        s = silver[silver["id"] == sid].sort_values("date")
        train = s[s["date"] < cutoff_date]
        test = s[s["date"] >= cutoff_date]

        train_sales = train["sales"].values.astype(np.float64)
        test_sales = test["sales"].values.astype(np.float64)

        # Generate forecasts
        snaive = seasonal_naive_forecast(train_sales, horizon=test_days, season=season)
        ma = moving_average_forecast(train_sales, horizon=test_days, window=ma_window)

        # Build result rows
        for i, (_, row) in enumerate(test.iterrows()):
            results.append({
                "id": sid,
                "item_id": row["item_id"],
                "cat_id": row["cat_id"],
                "store_id": row["store_id"],
                "state_id": row["state_id"],
                "date": row["date"],
                "actual": float(test_sales[i]),
                "snaive_forecast": float(snaive[i]),
                "ma_forecast": float(ma[i]),
                "train_sales": train_sales,  # for MASE computation
            })

    result_df = pd.DataFrame(results)
    print(f"\nBaseline results: {len(result_df)} rows "
          f"({result_df.id.nunique()} series × {test_days} days)")

    return result_df


def evaluate_baselines(
    baseline_df: pd.DataFrame,
    season: int = 7,
) -> Dict[str, pd.DataFrame]:
    """
    Evaluate baselines using WAPE, Bias, and MASE.

    Returns metrics at multiple levels:
    - overall
    - per category
    - per state (region)

    Parameters
    ----------
    baseline_df : pd.DataFrame
        Output of compute_baselines().

    Returns
    -------
    dict with keys: 'overall', 'by_category', 'by_state'
    """
    from src.metrics import wape, bias, mase

    def _eval_group(group_df: pd.DataFrame) -> Dict:
        """Evaluate one group (overall, or a category, or a state)."""
        metrics = {}
        for model_col, model_name in [("snaive_forecast", "snaive"), ("ma_forecast", "ma")]:
            actual_all = group_df["actual"].values
            forecast_all = group_df[model_col].values

            w = wape(actual_all, forecast_all)
            b = bias(actual_all, forecast_all)

            # MASE: compute per series, then average
            mase_vals = []
            for sid in group_df["id"].unique():
                s = group_df[group_df["id"] == sid]
                a = s["actual"].values
                f = s[model_col].values
                train_sales = s.iloc[0]["train_sales"]
                m = mase(a, f, train_sales, season=season)
                if np.isfinite(m):
                    mase_vals.append(m)

            avg_mase = np.mean(mase_vals) if mase_vals else np.inf

            metrics[f"{model_name}_wape"] = round(w, 4)
            metrics[f"{model_name}_bias"] = round(b, 4)
            metrics[f"{model_name}_mase"] = round(avg_mase, 4)

        return metrics

    # Overall
    overall = _eval_group(baseline_df)

    # By category
    by_cat = {}
    for cat in sorted(baseline_df["cat_id"].unique()):
        grp = baseline_df[baseline_df["cat_id"] == cat]
        by_cat[cat] = _eval_group(grp)

    # By state
    by_state = {}
    for state in sorted(baseline_df["state_id"].unique()):
        grp = baseline_df[baseline_df["state_id"] == state]
        by_state[state] = _eval_group(grp)

    return {
        "overall": overall,
        "by_category": by_cat,
        "by_state": by_state,
    }
