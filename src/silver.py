"""
Silver layer transformations.

Converts wide-format sales to long format, joins with calendar and prices,
and performs cleaning. This produces the analysis-ready dataset.

Databricks equivalent:
    # Unpivot with stack()
    sales_long = sales.selectExpr(
        "id", "item_id", "dept_id", "cat_id", "store_id", "state_id",
        "stack(1941, " + ", ".join([f"'d_{i}', d_{i}" for i in range(1,1942)]) + ") as (d, sales)"
    )
    # Join with calendar and prices using Delta
    silver = sales_long.join(calendar, on="d").join(prices, on=["store_id","item_id","wm_yr_wk"])
    silver.write.format("delta").mode("overwrite").save("silver/sales")
"""
import pandas as pd
import numpy as np
from pathlib import Path


def melt_sales(sales_wide: pd.DataFrame) -> pd.DataFrame:
    """
    Melt wide-format sales (one column per day) to long format.

    Input columns: id, item_id, dept_id, cat_id, store_id, state_id, d_1..d_1941
    Output columns: id, item_id, dept_id, cat_id, store_id, state_id, d, sales
    """
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in sales_wide.columns if c.startswith("d_")]

    print(f"  Melting {len(sales_wide)} series × {len(day_cols)} days ...")
    long = sales_wide.melt(
        id_vars=id_cols,
        value_vars=day_cols,
        var_name="d",
        value_name="sales",
    )
    # Convert sales to int16 to save memory (max daily sales per SKU-store << 32767)
    long["sales"] = long["sales"].astype(np.int16)
    print(f"  Melted shape: {long.shape}")
    print(f"  Memory: {long.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    return long


def join_calendar(sales_long: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """
    Join melted sales with calendar on the 'd' column.
    Adds date, weekday, month, year, events, SNAP flags.
    """
    print(f"  Joining with calendar ({len(calendar)} rows) ...")
    merged = sales_long.merge(calendar, on="d", how="left")
    print(f"  After calendar join: {merged.shape}")

    # Verify no rows lost
    assert len(merged) == len(sales_long), \
        f"Row count mismatch after calendar join: {len(merged)} vs {len(sales_long)}"

    return merged


def join_prices(
    sales_cal: pd.DataFrame,
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join sales+calendar with prices on store_id, item_id, wm_yr_wk.
    Adds sell_price column.
    """
    print(f"  Joining with prices ({len(prices)} rows) ...")
    merged = sales_cal.merge(
        prices,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
    )
    print(f"  After price join: {merged.shape}")

    # Some early days may not have prices (item not yet sold)
    n_missing = merged["sell_price"].isna().sum()
    pct_missing = n_missing / len(merged) * 100
    print(f"  Missing prices: {n_missing:,} ({pct_missing:.1f}%)")

    return merged


def compute_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Add revenue = sales * sell_price."""
    df = df.copy()
    df["revenue"] = df["sales"] * df["sell_price"]
    return df


def add_snap_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a single snap_eligible column based on the store's state.
    SNAP (Supplemental Nutrition Assistance Program) days vary by state.
    """
    df = df.copy()
    conditions = [
        (df["state_id"] == "CA") & (df["snap_CA"] == 1),
        (df["state_id"] == "TX") & (df["snap_TX"] == 1),
        (df["state_id"] == "WI") & (df["snap_WI"] == 1),
    ]
    df["snap_eligible"] = np.select(conditions, [1, 1, 1], default=0).astype(np.int8)
    return df


def build_silver(
    sales_parquet: str,
    calendar_parquet: str,
    prices_parquet: str,
    out_path: str,
) -> pd.DataFrame:
    """
    Full silver pipeline: melt → join calendar → join prices → clean → write.
    """
    print("=" * 60)
    print("SILVER LAYER BUILD")
    print("=" * 60)

    # Load bronze
    sales_wide = pd.read_parquet(sales_parquet)
    calendar = pd.read_parquet(calendar_parquet)
    prices = pd.read_parquet(prices_parquet)

    # Melt
    long = melt_sales(sales_wide)
    del sales_wide

    # Join calendar
    silver = join_calendar(long, calendar)
    del long

    # Join prices
    silver = join_prices(silver, prices)

    # Add derived columns
    silver = compute_revenue(silver)
    silver = add_snap_flag(silver)

    # Sort for time-series consistency
    silver = silver.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)

    # Select and order final columns
    keep_cols = [
        "id", "item_id", "dept_id", "cat_id", "store_id", "state_id",
        "d", "date", "wm_yr_wk",
        "sales", "sell_price", "revenue",
        "weekday", "wday", "month", "year",
        "event_name_1", "event_type_1", "event_name_2", "event_type_2",
        "snap_eligible",
    ]
    silver = silver[keep_cols]

    # Write
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    silver.to_parquet(out_path, index=False, engine="pyarrow")
    size_mb = Path(out_path).stat().st_size / (1024 ** 2)
    print(f"\n  Silver written to {out_path} ({size_mb:.1f} MB)")
    print(f"  Final shape: {silver.shape}")
    print(f"  Memory: {silver.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    return silver
