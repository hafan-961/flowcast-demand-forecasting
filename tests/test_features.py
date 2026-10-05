"""
Tests for feature engineering, primarily focusing on data leakage.
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.features import generate_features, assert_no_leakage

def test_no_leakage():
    # Create dummy data for 1 series over 60 days
    dates = pd.date_range("2026-01-01", periods=60)
    sales = np.arange(60) # 0 to 59
    df = pd.DataFrame({
        "id": ["item1_store1"] * 60,
        "date": dates,
        "sales": sales,
        "sell_price": [1.0] * 60,
        "item_id": ["item1"] * 60,
        "dept_id": ["dept1"] * 60,
        "cat_id": ["cat1"] * 60,
        "store_id": ["store1"] * 60,
        "state_id": ["state1"] * 60,
        "event_name_1": [None] * 60
    })
    
    horizon = 28
    feat_df = generate_features(df, horizon=horizon)
    
    # Check that lag_28 at index 28 (which is the 29th day) is equal to sales at index 0
    # sales[0] = 0
    assert feat_df.loc[28, "lag_28"] == 0
    
    # sales[28] is 28. lag_28 should not be 28.
    assert feat_df.loc[28, "lag_28"] != 28
    
    # Run the strict leakage assertion function
    assert_no_leakage(feat_df, "sales", ["lag_28"], horizon=horizon)
    
    # Verify rolling mean
    # rolling_mean_7_lag_28 at index 34 (which corresponds to target at index 34).
    # lag 28 of 34 is 6. The 7 days ending at index 6 are indices 0, 1, 2, 3, 4, 5, 6
    # sales are 0, 1, 2, 3, 4, 5, 6. Mean is 3.0
    assert feat_df.loc[34, "rolling_mean_7_lag_28"] == 3.0
