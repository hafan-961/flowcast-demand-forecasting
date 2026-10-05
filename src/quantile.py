"""
Quantile model training using LightGBM.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import mlflow

from src.features import generate_features
from src.model import time_split
from src.metrics import prediction_interval_coverage

def train_lgb_quantile(
    df: pd.DataFrame, 
    features: list, 
    target: str, 
    lgb_params: dict,
    alpha: float = 0.90,
    test_days: int = 28,
):
    """
    Train a LightGBM quantile regression model.
    """
    params = lgb_params.copy()
    params["objective"] = "quantile"
    params["alpha"] = alpha
    
    # Feature Engineering
    df_feat = generate_features(df, horizon=test_days)
    df_feat = df_feat.dropna(subset=["lag_28"]).reset_index(drop=True)
    
    # Time-based split
    train, test = time_split(df_feat, test_days=test_days)
    
    train_dataset = lgb.Dataset(train[features], label=train[target])
    
    # Train
    print(f"\\nTraining Quantile Model (alpha={alpha})...")
    model = lgb.train(
        params,
        train_dataset,
        num_boost_round=params.get("n_estimators", 100),
    )
    
    # Predict
    test_preds = model.predict(test[features])
    test_preds = np.clip(test_preds, 0, None)  # demand cannot be negative
    
    test_results = test.copy()
    test_results[f"p{int(alpha*100)}_forecast"] = test_preds
    
    # Coverage
    coverage = prediction_interval_coverage(test_results[target].values, test_preds)
    print(f"Prediction Interval Coverage (P{int(alpha*100)}): {coverage:.1%}")
    
    # Log to MLflow
    with mlflow.start_run(run_name=f"lgbm_quantile_p{int(alpha*100)}") as run:
        mlflow.log_params(params)
        mlflow.log_metric("test_coverage", coverage)
        mlflow.lightgbm.log_model(
            lgb_model=model, 
            artifact_path="model",
            registered_model_name=f"FlowCast_LightGBM_P{int(alpha*100)}"
        )
        
    return test_results, model
