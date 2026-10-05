"""
Bronze layer ingestion.

Reads raw M5 CSVs, applies the subset filter, and writes Parquet files
to data/bronze/. This is the "raw but typed" layer in the medallion architecture.

Databricks equivalent:
    spark.read.csv("datasets/sales_train_evaluation.csv", header=True, inferSchema=True)
         .write.format("delta").mode("overwrite").save("bronze/sales")
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List


def ingest_sales_bronze(
    raw_path: str,
    out_path: str,
    selected_items: List[str],
) -> pd.DataFrame:
    """
    Read wide-format sales CSV, filter to selected items,
    and write to Parquet.

    Returns the filtered dataframe for inspection.
    """
    print(f"Reading {raw_path} ...")
    sales = pd.read_csv(raw_path)
    print(f"  Raw shape: {sales.shape}")

    # Filter to subset
    sales_sub = sales[sales["item_id"].isin(selected_items)].copy()
    print(f"  Subset shape: {sales_sub.shape}")

    # Write to parquet
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sales_sub.to_parquet(out_path, index=False, engine="pyarrow")
    size_mb = Path(out_path).stat().st_size / (1024 ** 2)
    print(f"  Written to {out_path} ({size_mb:.1f} MB)")

    return sales_sub


def ingest_calendar_bronze(raw_path: str, out_path: str) -> pd.DataFrame:
    """Read calendar CSV and write to Parquet."""
    print(f"Reading {raw_path} ...")
    cal = pd.read_csv(raw_path)
    print(f"  Shape: {cal.shape}")

    # Parse date column
    cal["date"] = pd.to_datetime(cal["date"])

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cal.to_parquet(out_path, index=False, engine="pyarrow")
    size_mb = Path(out_path).stat().st_size / (1024 ** 2)
    print(f"  Written to {out_path} ({size_mb:.1f} MB)")

    return cal


def ingest_prices_bronze(
    raw_path: str,
    out_path: str,
    selected_items: List[str],
) -> pd.DataFrame:
    """Read sell_prices CSV, filter to selected items, write Parquet."""
    print(f"Reading {raw_path} ...")
    prices = pd.read_csv(raw_path)
    print(f"  Raw shape: {prices.shape}")

    prices_sub = prices[prices["item_id"].isin(selected_items)].copy()
    print(f"  Subset shape: {prices_sub.shape}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    prices_sub.to_parquet(out_path, index=False, engine="pyarrow")
    size_mb = Path(out_path).stat().st_size / (1024 ** 2)
    print(f"  Written to {out_path} ({size_mb:.1f} MB)")

    return prices_sub
