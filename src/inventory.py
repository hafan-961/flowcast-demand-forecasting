"""
Inventory management metrics.

Computes Safety Stock, Reorder Point (ROP), and flags stockouts/overstocks.

FORMULAS:
- Lead Time Demand (LTD) = daily_forecast * lead_time_days
- Safety Stock (SS) = z_score * sqrt(lead_time_days * variance_of_demand)
  (Variance of demand is estimated from the rolling 28-day historical std dev).
- Reorder Point (ROP) = LTD + SS

FLAGS:
- Stockout Flag: actual > ROP (demand exceeded our maximum expected inventory position)
- Overstock Flag: actual == 0 AND ROP > (3 * mean_sales) 
  (we provisioned high inventory but sold nothing)
"""
import pandas as pd
import numpy as np

def compute_inventory_metrics(
    df: pd.DataFrame,
    forecast_col: str,
    lead_time_days: int,
    z_score: float,
) -> pd.DataFrame:
    """
    Compute inventory recommendations and flags.
    
    Parameters
    ----------
    df : pd.DataFrame
        Test set results with actuals, forecast, and rolling_std.
    forecast_col : str
        Column name for the point forecast (e.g., 'lgbm_forecast').
    lead_time_days : int
        Assumed replenishment lead time.
    z_score : float
        Z-score for target service level (e.g. 1.645 for 95%).
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added inventory columns.
    """
    res = df.copy()
    
    # Ensure no negative forecasts or std devs
    forecast = np.clip(res[forecast_col].values, 0, None)
    std_dev = np.clip(res["rolling_std_28_lag_28"].fillna(0).values, 0, None)
    actual = res["sales"].values
    
    # Formulas
    # 1. Lead Time Demand
    res["lead_time_demand"] = forecast * lead_time_days
    
    # 2. Safety Stock = Z * std_dev * sqrt(L)
    res["safety_stock"] = z_score * std_dev * np.sqrt(lead_time_days)
    
    # 3. Reorder Point
    res["reorder_point"] = res["lead_time_demand"] + res["safety_stock"]
    
    # Flags
    # Stockout: actual demand > Reorder Point
    res["stockout_flag"] = (actual > res["reorder_point"]).astype(int)
    
    # Overstock: provisioned for a lot, but sold zero
    # Define "a lot" as ROP > 3 units (heuristic for this dataset's scale)
    res["overstock_flag"] = ((actual == 0) & (res["reorder_point"] > 3)).astype(int)
    
    return res
