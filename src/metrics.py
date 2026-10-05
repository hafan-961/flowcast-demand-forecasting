"""
Forecast evaluation metrics.

All formulas are documented here and in the README.

WAPE (Weighted Absolute Percentage Error):
    WAPE = Σ|actual - forecast| / Σ|actual|
    - Scale-independent, handles zeros better than MAPE
    - Lower is better; 0 = perfect

Bias (Normalized):
    Bias = Σ(forecast - actual) / Σ|actual|
    - Positive = over-forecast, Negative = under-forecast
    - Closer to 0 is better

MASE (Mean Absolute Scaled Error):
    MASE = mean(|actual - forecast|) / mean(|actual_t - actual_{t-season}|)
    - Scaled by the in-sample seasonal naive error
    - MASE < 1 means the model beats the seasonal naive baseline
    - Season = 7 for daily data (weekly cycle)
"""
import numpy as np
from typing import Optional


def wape(actual: np.ndarray, forecast: np.ndarray) -> float:
    """
    Weighted Absolute Percentage Error.

    WAPE = Σ|actual - forecast| / Σ|actual|

    Parameters
    ----------
    actual : array-like, shape (n,)
    forecast : array-like, shape (n,)

    Returns
    -------
    float
        WAPE value. Returns np.inf if sum of actuals is 0.
    """
    actual = np.asarray(actual, dtype=np.float64)
    forecast = np.asarray(forecast, dtype=np.float64)
    denom = np.sum(np.abs(actual))
    if denom == 0:
        return np.inf
    return float(np.sum(np.abs(actual - forecast)) / denom)


def bias(actual: np.ndarray, forecast: np.ndarray) -> float:
    """
    Normalized Bias.

    Bias = Σ(forecast - actual) / Σ|actual|

    Positive = over-forecast, Negative = under-forecast.

    Parameters
    ----------
    actual : array-like, shape (n,)
    forecast : array-like, shape (n,)

    Returns
    -------
    float
        Bias value. Returns np.inf if sum of actuals is 0.
    """
    actual = np.asarray(actual, dtype=np.float64)
    forecast = np.asarray(forecast, dtype=np.float64)
    denom = np.sum(np.abs(actual))
    if denom == 0:
        return np.inf
    return float(np.sum(forecast - actual) / denom)


def mase(
    actual: np.ndarray,
    forecast: np.ndarray,
    insample: np.ndarray,
    season: int = 7,
) -> float:
    """
    Mean Absolute Scaled Error.

    MASE = mean(|actual - forecast|) / mean(|insample_t - insample_{t-season}|)

    Parameters
    ----------
    actual : array-like, shape (n,)
        Test-period actuals.
    forecast : array-like, shape (n,)
        Test-period forecasts.
    insample : array-like, shape (m,)
        Training-period actuals for computing the naive scaling factor.
        Must have length > season.
    season : int
        Seasonal period (7 for daily data with weekly cycle).

    Returns
    -------
    float
        MASE value. Returns np.inf if the naive error is 0.
    """
    actual = np.asarray(actual, dtype=np.float64)
    forecast = np.asarray(forecast, dtype=np.float64)
    insample = np.asarray(insample, dtype=np.float64)

    # Numerator: MAE of the forecast
    mae_forecast = np.mean(np.abs(actual - forecast))

    # Denominator: MAE of the in-sample seasonal naive
    naive_errors = np.abs(insample[season:] - insample[:-season])
    mae_naive = np.mean(naive_errors)

    if mae_naive == 0:
        return np.inf

    return float(mae_forecast / mae_naive)


def prediction_interval_coverage(
    actual: np.ndarray,
    upper_bound: np.ndarray,
) -> float:
    """
    Fraction of actuals that fall at or below the upper bound (e.g. P90).

    Parameters
    ----------
    actual : array-like
    upper_bound : array-like
        Upper quantile forecast (e.g. P90).

    Returns
    -------
    float
        Coverage fraction (ideally close to the target quantile).
    """
    actual = np.asarray(actual, dtype=np.float64)
    upper_bound = np.asarray(upper_bound, dtype=np.float64)
    return float(np.mean(actual <= upper_bound))
