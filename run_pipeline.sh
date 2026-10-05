#!/bin/bash
set -e

echo "========================================="
echo " 🚀 Starting FlowCast Pipeline 🚀 "
echo "========================================="

echo "\n[1/9] Running Bronze Ingestion..."
python3 src/bronze.py

echo "\n[2/9] Running Silver Cleaning..."
python3 src/silver.py

echo "\n[3/9] Running EDA..."
python3 src/eda.py

echo "\n[4/9] Running Baselines..."
python3 src/baselines.py

echo "\n[5/9] Engineering Features..."
python3 src/features.py

echo "\n[6/9] Training P50 Forecast (MLflow)..."
python3 src/model.py

echo "\n[7/9] Training P90 Quantile Forecast (MLflow)..."
python3 src/quantile.py

echo "\n[8/9] Calculating Inventory/Reorder Points (Gold)..."
python3 src/inventory.py

echo "\n[9/9] Running PuLP Optimizer vs Naive Allocation..."
python3 scripts/run_optimizer.py

echo "\n========================================="
echo " ✅ Pipeline Complete! ✅ "
echo "========================================="
echo "Check data/gold/ for your final outputs."
echo "Run 'mlflow ui --backend-store-uri sqlite:///mlruns.db' to see your models."
