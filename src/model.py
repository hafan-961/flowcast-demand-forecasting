"""
LightGBM training, Rolling-Origin CV, and MLflow tracking.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import matplotlib.pyplot as plt
from pathlib import Path

from src.metrics import wape, bias, mase
from src.features import generate_features

def time_split(df: pd.DataFrame, test_days: int = 28):
    """
    Split the dataset into train and test sets strictly by time.
    """
    all_dates = sorted(df["date"].unique())
    cutoff_date = all_dates[-test_days]
    
    train = df[df["date"] < cutoff_date].copy()
    test = df[df["date"] >= cutoff_date].copy()
    
    return train, test

def train_lgb_cv(
    df: pd.DataFrame, 
    features: list, 
    target: str, 
    lgb_params: dict,
    test_days: int = 28,
    cv_folds: int = 3,
    cv_step: int = 28,
    experiment_name: str = "flowcast-lgbm",
    season: int = 7
):
    """
    Train LightGBM with rolling-origin CV and log to MLflow.
    """
    mlflow.set_experiment(experiment_name)
    
    # 1. Feature Engineering
    df_feat = generate_features(df, horizon=test_days)
    
    # Drop rows where lag_28 is NaN (first 28 days)
    df_feat = df_feat.dropna(subset=["lag_28"]).reset_index(drop=True)
    
    # 2. Time-based split for true holdout test set
    train_full, test = time_split(df_feat, test_days=test_days)
    
    print(f"Full Train data shape: {train_full.shape}")
    print(f"Test data shape: {test.shape}")
    
    all_train_dates = sorted(train_full["date"].unique())
    
    fold_metrics = []
    best_models = []
    
    with mlflow.start_run(run_name="lgbm_cv_run") as run:
        mlflow.log_params(lgb_params)
        mlflow.log_param("features", features)
        mlflow.log_param("cv_folds", cv_folds)
        
        # 3. Rolling-Origin CV on train_full
        for fold in range(cv_folds):
            # Val is the last cv_step days of the current train_dates
            # Train is everything before that
            val_end_idx = len(all_train_dates) - fold * cv_step
            val_start_idx = val_end_idx - cv_step
            
            val_start_date = all_train_dates[val_start_idx]
            val_end_date = all_train_dates[val_end_idx - 1]
            
            fold_train = train_full[train_full["date"] < val_start_date]
            fold_val = train_full[(train_full["date"] >= val_start_date) & (train_full["date"] <= val_end_date)]
            
            print(f"\\nFold {fold+1}:")
            print(f"  Train: <= {fold_train['date'].max().date()} ({len(fold_train)} rows)")
            print(f"  Val:   {val_start_date.date()} to {val_end_date.date()} ({len(fold_val)} rows)")
            
            train_dataset = lgb.Dataset(fold_train[features], label=fold_train[target])
            val_dataset = lgb.Dataset(fold_val[features], label=fold_val[target], reference=train_dataset)
            
            # Train
            model = lgb.train(
                lgb_params,
                train_dataset,
                valid_sets=[train_dataset, val_dataset],
                valid_names=["train", "val"],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=0)
                ]
            )
            best_models.append(model)
            
            # Predict and evaluate on Validation
            preds = model.predict(fold_val[features])
            
            # Clip negative predictions to 0
            preds = np.clip(preds, 0, None)
            
            fold_wape = wape(fold_val[target].values, preds)
            fold_bias = bias(fold_val[target].values, preds)
            
            print(f"  Val WAPE: {fold_wape:.4f} | Val Bias: {fold_bias:.4f}")
            mlflow.log_metric(f"fold_{fold+1}_val_wape", fold_wape)
            mlflow.log_metric(f"fold_{fold+1}_val_bias", fold_bias)
            fold_metrics.append((fold_wape, fold_bias))
            
        mean_cv_wape = np.mean([m[0] for m in fold_metrics])
        mlflow.log_metric("mean_cv_wape", mean_cv_wape)
        print(f"\\nMean CV WAPE: {mean_cv_wape:.4f}")
        
        # 4. Final Evaluation on Test Set using the model from the last fold
        # In a real setup, we might retrain on all train_full, but for now we'll use Fold 1's model 
        # (which trained on the most recent data)
        final_model = best_models[0]
        
        test_preds = final_model.predict(test[features])
        test_preds = np.clip(test_preds, 0, None)
        
        test_wape = wape(test[target].values, test_preds)
        test_bias = bias(test[target].values, test_preds)
        
        print(f"\\n--- TEST SET EVALUATION ---")
        print(f"Test WAPE: {test_wape:.4f}")
        print(f"Test Bias: {test_bias:.4f}")
        
        mlflow.log_metric("test_wape", test_wape)
        mlflow.log_metric("test_bias", test_bias)
        
        # Feature Importance Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        lgb.plot_importance(final_model, ax=ax, max_num_features=15, importance_type='gain')
        plt.tight_layout()
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png")
        
        # Log the model
        mlflow.lightgbm.log_model(
            lgb_model=final_model, 
            artifact_path="model",
            registered_model_name="FlowCast_LightGBM_P50"
        )
        
        # Save predictions for analysis
        test_results = test.copy()
        test_results["lgbm_forecast"] = test_preds
        test_results.to_parquet("data/silver/lgbm_results.parquet", index=False)
        
        return test_results, final_model
