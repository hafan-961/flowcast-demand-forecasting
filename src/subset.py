"""
Subset selection utility.

Selects a stratified subset of top-selling items from the M5 dataset
to keep memory usage manageable on limited hardware.
"""
import pandas as pd
from typing import Dict, List


def select_top_items(
    sales: pd.DataFrame,
    items_per_category: Dict[str, int],
    lookback_days: int = 365,
) -> List[str]:
    """
    Select top-selling items per category by total sales volume
    over the last `lookback_days`.

    Parameters
    ----------
    sales : pd.DataFrame
        Full sales_train_evaluation dataframe (wide format).
    items_per_category : dict
        e.g. {"FOODS": 60, "HOUSEHOLD": 25, "HOBBIES": 15}
    lookback_days : int
        Number of trailing days to compute total sales.

    Returns
    -------
    list of str
        Selected item_id values.
    """
    day_cols = [c for c in sales.columns if c.startswith("d_")]
    last_n = day_cols[-lookback_days:]
    sales = sales.copy()
    sales["_total"] = sales[last_n].sum(axis=1)

    item_cat = sales[["item_id", "cat_id"]].drop_duplicates()
    item_totals = (
        sales.groupby("item_id")["_total"]
        .sum()
        .sort_values(ascending=False)
    )

    selected: List[str] = []
    for cat, n in items_per_category.items():
        cat_items = item_cat.loc[item_cat.cat_id == cat, "item_id"]
        top_n = item_totals[item_totals.index.isin(cat_items)].head(n).index.tolist()
        selected.extend(top_n)

    return selected
