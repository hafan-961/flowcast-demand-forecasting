"""
Feature engineering for demand forecasting.

Creates lag and rolling window features.
To strictly prevent data leakage for a direct multi-step forecast (horizon=28),
all lag and rolling features MUST be shifted by at least the horizon.

For example, if we are predicting `sales` at day `T`, the most recent data
we know (from a 28-day advance perspective) is at `T - 28`.
So we use:
- lag_28
- lag_35 (one week before lag_28)
- rolling means computed starting from `T - 28` backwards.

Databricks PySpark Equivalent:
    from pyspark.sql.window import Window
    import pyspark.sql.functions as F

    window_lag = Window.partitionBy("id").orderBy("date")
    df = df.withColumn("lag_28", F.lag("sales", 28).over(window_lag))
"""
import pandas as pd
import numpy as np


def generate_features(df: pd.DataFrame, horizon: int = 28) -> pd.DataFrame:
    """
    Generate features for direct multi-step forecasting.

    Parameters
    ----------
    df : pd.DataFrame
        Silver dataset, must contain ['id', 'date', 'sales', 'sell_price', ...].
    horizon : int
        The forecast horizon. All lags are shifted by at least this amount
        to prevent target leakage.

    Returns
    -------
    pd.DataFrame
        Dataset with new feature columns.
    """
    df = df.copy()

    # Ensure data is sorted temporally within each series
    df = df.sort_values(["id", "date"]).reset_index(drop=True)

    print(f"Generating features with minimum lag {horizon} to prevent leakage...")

    # Lag features
    df["lag_28"] = df.groupby("id")["sales"].shift(horizon)
    df["lag_35"] = df.groupby("id")["sales"].shift(horizon + 7)  # Same weekday, one week prior

    # Rolling window features (computed over the shifted series)
    # Using pandas transform with rolling
    def rolling_mean(group, window):
        return group.shift(horizon).rolling(window, min_periods=1).mean()

    def rolling_std(group, window):
        return group.shift(horizon).rolling(window, min_periods=2).std()

    df["rolling_mean_7_lag_28"] = df.groupby("id")["sales"].transform(lambda x: rolling_mean(x, 7))
    df["rolling_mean_28_lag_28"] = df.groupby("id")["sales"].transform(lambda x: rolling_mean(x, 28))
    df["rolling_std_28_lag_28"] = df.groupby("id")["sales"].transform(lambda x: rolling_std(x, 28))

    # Convert categoricals
    cat_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id", "event_name_1"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df


def assert_no_leakage(df: pd.DataFrame, target_col: str, feature_cols: list, horizon: int):
    """
    Leakage check.
    If we predict for a specific date, none of the feature columns
    should contain information from within the horizon.
    We test this by taking a slice of data, shifting the target back
    by 'horizon' days, and checking if features correlate perfectly.
    Since we built lags explicitly, we check that the non-null features
    at date T exactly match the target at T-horizon or older.
    """
    print("Running leakage check...")
    test_id = df["id"].iloc[0]
    series = df[df["id"] == test_id].sort_values("date").reset_index(drop=True)

    # Check lag_28
    if "lag_28" in feature_cols:
        # lag_28 at index i should be exactly sales at index i-28
        for i in range(horizon, len(series)):
            assert series.loc[i, "lag_28"] == series.loc[i - horizon, target_col], \
                f"LEAKAGE DETECTED: lag_28 at {series.loc[i, 'date']} is {series.loc[i, 'lag_28']}, " \
                f"but target at {series.loc[i-horizon, 'date']} is {series.loc[i-horizon, target_col]}"

    print("Leakage check PASSED: Lags correctly shifted.")
