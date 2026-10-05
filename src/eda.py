"""
EDA and Data Quality Report — Phase 1.

Produces summary statistics, distribution plots, and a data quality
assessment for the silver dataset.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path


def run_eda(silver_path: str, output_dir: str) -> dict:
    """
    Run full EDA on the silver dataset.

    Returns a dict of quality metrics.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading silver data ...")
    df = pd.read_parquet(silver_path)
    print(f"Shape: {df.shape}")

    quality = {}

    # ── 1. Basic stats ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("1. BASIC STATISTICS")
    print("=" * 60)

    print(f"\nDate range: {df.date.min().date()} to {df.date.max().date()}")
    print(f"Total rows: {len(df):,}")
    print(f"Unique series: {df.id.nunique()}")
    print(f"Unique items: {df.item_id.nunique()}")
    print(f"Unique stores: {df.store_id.nunique()}")

    quality["total_rows"] = len(df)
    quality["date_range"] = f"{df.date.min().date()} to {df.date.max().date()}"

    # ── 2. Missing values ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("2. MISSING VALUES")
    print("=" * 60)

    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_report = pd.DataFrame({"count": missing, "pct": missing_pct})
    missing_report = missing_report[missing_report["count"] > 0]
    print(missing_report.to_string())

    quality["missing_sell_price_pct"] = float(missing_pct.get("sell_price", 0))

    # ── 3. Sales distribution ───────────────────────────────────
    print("\n" + "=" * 60)
    print("3. SALES DISTRIBUTION")
    print("=" * 60)

    print(df["sales"].describe().to_string())
    zero_pct = (df["sales"] == 0).mean() * 100
    print(f"\nZero-sales days: {zero_pct:.1f}%")
    quality["zero_sales_pct"] = round(zero_pct, 1)

    # Sales distribution plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram of non-zero sales
    nonzero = df.loc[df["sales"] > 0, "sales"]
    axes[0].hist(nonzero, bins=100, color="#4A90D9", edgecolor="white", alpha=0.8)
    axes[0].set_xlabel("Daily Sales (units)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution of Non-Zero Daily Sales")
    axes[0].set_xlim(0, nonzero.quantile(0.99))

    # Zero fraction by category
    zero_by_cat = df.groupby("cat_id").apply(
        lambda x: (x["sales"] == 0).mean() * 100
    ).sort_values()
    axes[1].barh(zero_by_cat.index, zero_by_cat.values, color=["#E8764D", "#4A90D9", "#50C878"])
    axes[1].set_xlabel("Zero-Sales Days (%)")
    axes[1].set_title("Intermittent Demand by Category")
    for i, v in enumerate(zero_by_cat.values):
        axes[1].text(v + 0.5, i, f"{v:.1f}%", va="center")

    plt.tight_layout()
    plt.savefig(out / "sales_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out / 'sales_distribution.png'}")

    # Zero fraction by category (print)
    print("\nZero-sales % by category:")
    for cat, val in zero_by_cat.items():
        print(f"  {cat}: {val:.1f}%")

    # ── 4. Time series aggregates ───────────────────────────────
    print("\n" + "=" * 60)
    print("4. TIME SERIES PATTERNS")
    print("=" * 60)

    # Daily total sales over time
    daily_total = df.groupby("date")["sales"].sum()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Total daily sales
    axes[0, 0].plot(daily_total.index, daily_total.values, linewidth=0.5, color="#4A90D9")
    axes[0, 0].set_title("Total Daily Sales (All SKUs, All Stores)")
    axes[0, 0].set_ylabel("Units")
    axes[0, 0].xaxis.set_major_locator(mdates.YearLocator())
    axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # By state
    for state in ["CA", "TX", "WI"]:
        state_daily = df[df.state_id == state].groupby("date")["sales"].sum()
        axes[0, 1].plot(state_daily.index, state_daily.values, linewidth=0.5, label=state)
    axes[0, 1].set_title("Daily Sales by State (Region)")
    axes[0, 1].set_ylabel("Units")
    axes[0, 1].legend()
    axes[0, 1].xaxis.set_major_locator(mdates.YearLocator())
    axes[0, 1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # By category
    for cat in ["FOODS", "HOUSEHOLD", "HOBBIES"]:
        cat_daily = df[df.cat_id == cat].groupby("date")["sales"].sum()
        axes[1, 0].plot(cat_daily.index, cat_daily.values, linewidth=0.5, label=cat)
    axes[1, 0].set_title("Daily Sales by Category")
    axes[1, 0].set_ylabel("Units")
    axes[1, 0].legend()
    axes[1, 0].xaxis.set_major_locator(mdates.YearLocator())
    axes[1, 0].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Day-of-week seasonality
    dow_sales = df.groupby("wday")["sales"].mean()
    day_names = ["Sat", "Sun", "Mon", "Tue", "Wed", "Thu", "Fri"]
    axes[1, 1].bar(range(1, 8), [dow_sales.get(i, 0) for i in range(1, 8)],
                   color="#4A90D9", edgecolor="white")
    axes[1, 1].set_xticks(range(1, 8))
    axes[1, 1].set_xticklabels(day_names)
    axes[1, 1].set_title("Average Sales by Day of Week")
    axes[1, 1].set_ylabel("Mean Units/Series/Day")

    plt.tight_layout()
    plt.savefig(out / "time_series_patterns.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out / 'time_series_patterns.png'}")

    # Print DOW stats
    print("\nAverage sales by day of week:")
    for i, name in enumerate(day_names, 1):
        print(f"  {name}: {dow_sales.get(i, 0):.2f}")

    # ── 5. Sales by store ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("5. SALES BY STORE")
    print("=" * 60)

    store_stats = df.groupby("store_id")["sales"].agg(["mean", "std", "sum"])
    store_stats = store_stats.sort_values("sum", ascending=False)
    print(store_stats.to_string())

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(store_stats.index, store_stats["sum"] / 1e6, color="#4A90D9", edgecolor="white")
    ax.set_ylabel("Total Sales (millions)")
    ax.set_title("Total Sales by Store (Fulfilment Node)")
    ax.set_xlabel("Store ID")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out / "sales_by_store.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out / 'sales_by_store.png'}")

    # ── 6. Price analysis ───────────────────────────────────────
    print("\n" + "=" * 60)
    print("6. PRICE ANALYSIS")
    print("=" * 60)

    prices_valid = df.dropna(subset=["sell_price"])
    print(f"Price stats (where available):")
    print(prices_valid["sell_price"].describe().to_string())

    # Price changes over time (avg across items)
    weekly_price = prices_valid.groupby("wm_yr_wk")["sell_price"].mean()
    quality["avg_price"] = round(float(prices_valid["sell_price"].mean()), 2)

    # ── 7. Event impact ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("7. EVENT IMPACT")
    print("=" * 60)

    event_days = df[df.event_name_1.notna()]
    non_event_days = df[df.event_name_1.isna()]
    print(f"Event days: {event_days.date.nunique()} unique dates")
    print(f"Non-event days: {non_event_days.date.nunique()} unique dates")
    print(f"Avg sales on event days: {event_days.sales.mean():.2f}")
    print(f"Avg sales on non-event days: {non_event_days.sales.mean():.2f}")
    print(f"Lift: {(event_days.sales.mean() / non_event_days.sales.mean() - 1) * 100:.1f}%")

    quality["event_lift_pct"] = round(
        (event_days.sales.mean() / non_event_days.sales.mean() - 1) * 100, 1
    )

    # Event type breakdown
    print("\nAvg sales by event type:")
    event_type_sales = event_days.groupby("event_type_1")["sales"].mean().sort_values(ascending=False)
    print(event_type_sales.to_string())

    # ── 8. SNAP impact ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("8. SNAP IMPACT")
    print("=" * 60)

    snap_sales = df.groupby("snap_eligible")["sales"].mean()
    print(f"Avg sales on SNAP-eligible days: {snap_sales.get(1, 0):.2f}")
    print(f"Avg sales on non-SNAP days: {snap_sales.get(0, 0):.2f}")
    if snap_sales.get(0, 0) > 0:
        print(f"SNAP lift: {(snap_sales.get(1, 0) / snap_sales.get(0, 0) - 1) * 100:.1f}%")

    # ── 9. Intermittent demand analysis ─────────────────────────
    print("\n" + "=" * 60)
    print("9. INTERMITTENT DEMAND ANALYSIS")
    print("=" * 60)

    # Calculate ADI (Average Demand Interval) and CV² for each series
    series_stats = []
    for sid, grp in df.groupby("id"):
        sales_arr = grp["sales"].values
        nonzero_idx = np.where(sales_arr > 0)[0]
        if len(nonzero_idx) > 1:
            intervals = np.diff(nonzero_idx)
            adi = intervals.mean()
            cv2 = (sales_arr[sales_arr > 0].std() / sales_arr[sales_arr > 0].mean()) ** 2
        elif len(nonzero_idx) == 1:
            adi = len(sales_arr)
            cv2 = 0
        else:
            adi = np.inf
            cv2 = 0
        series_stats.append({
            "id": sid,
            "adi": adi,
            "cv2": cv2,
            "zero_pct": (sales_arr == 0).mean() * 100,
            "mean_sales": sales_arr.mean(),
        })

    stats_df = pd.DataFrame(series_stats)

    # Classify demand (Syntetos-Boylan)
    # Smooth: ADI < 1.32 and CV² < 0.49
    # Erratic: ADI < 1.32 and CV² >= 0.49
    # Intermittent: ADI >= 1.32 and CV² < 0.49
    # Lumpy: ADI >= 1.32 and CV² >= 0.49
    def classify(row):
        if row["adi"] < 1.32:
            return "Smooth" if row["cv2"] < 0.49 else "Erratic"
        else:
            return "Intermittent" if row["cv2"] < 0.49 else "Lumpy"

    stats_df["demand_class"] = stats_df.apply(classify, axis=1)
    class_counts = stats_df["demand_class"].value_counts()
    print("Demand classification (Syntetos-Boylan):")
    for cls, cnt in class_counts.items():
        print(f"  {cls}: {cnt} series ({cnt/len(stats_df)*100:.1f}%)")

    quality["demand_classes"] = class_counts.to_dict()

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = {"Smooth": "#50C878", "Erratic": "#FFD700", "Intermittent": "#4A90D9", "Lumpy": "#E8764D"}
    for cls in ["Smooth", "Erratic", "Intermittent", "Lumpy"]:
        mask = stats_df["demand_class"] == cls
        ax.scatter(stats_df.loc[mask, "adi"], stats_df.loc[mask, "cv2"],
                   alpha=0.5, s=20, label=cls, color=colors.get(cls, "gray"))
    ax.axvline(1.32, color="gray", linestyle="--", alpha=0.5)
    ax.axhline(0.49, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("ADI (Average Demand Interval)")
    ax.set_ylabel("CV² (Coefficient of Variation²)")
    ax.set_title("Demand Classification (Syntetos-Boylan)")
    ax.legend()
    ax.set_xlim(0, min(stats_df.adi.quantile(0.99), 20))
    ax.set_ylim(0, min(stats_df.cv2.quantile(0.99), 5))
    plt.tight_layout()
    plt.savefig(out / "demand_classification.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out / 'demand_classification.png'}")

    # ── 10. Data Quality Summary ────────────────────────────────
    print("\n" + "=" * 60)
    print("10. DATA QUALITY SUMMARY")
    print("=" * 60)

    # Check for duplicates
    dupes = df.duplicated(subset=["store_id", "item_id", "date"]).sum()
    print(f"Duplicate (store, item, date) rows: {dupes}")
    quality["duplicates"] = int(dupes)

    # Check for negative sales
    neg = (df["sales"] < 0).sum()
    print(f"Negative sales: {neg}")
    quality["negative_sales"] = int(neg)

    # Check date continuity (any gaps?)
    dates = sorted(df.date.unique())
    expected_days = (pd.Timestamp(dates[-1]) - pd.Timestamp(dates[0])).days + 1
    actual_days = len(dates)
    print(f"Expected days: {expected_days}, Actual unique days: {actual_days}")
    print(f"Date gaps: {expected_days - actual_days}")
    quality["date_gaps"] = expected_days - actual_days

    # Series with constant zero
    all_zero = stats_df[stats_df.mean_sales == 0]
    print(f"Series with all-zero sales: {len(all_zero)}")
    quality["all_zero_series"] = len(all_zero)

    print("\n✅ EDA complete.")

    return quality
