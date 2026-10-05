# Resume Bullets for HP Data Science Intern Role

Here are a few variations of resume bullets you can use to highlight this project on your resume. They follow the **STAR** method (Situation, Task, Action, Result) and are heavily optimized for the "AI for Placement and Inventory Management" internship at HP.

### Option 1: The End-to-End Heavy Hitter (Best for primary project slot)
* **Developed an end-to-end supply chain optimization prototype** mimicking a Databricks Medallion architecture (Bronze/Silver/Gold) using Pandas and Parquet to process multi-regional point-of-sale data.
* **Engineered a LightGBM quantile regression model** with strictly leak-proof rolling-origin cross-validation, predicting SKU-level safety stock requirements with 90.6% target interval coverage.
* **Built a Linear Programming placement optimizer in PuLP (GLPK solver)** to dynamically allocate inventory across 10 constrained fulfilment nodes, reducing simulated weekly transfer and stockout costs by 7.1%.
* **Standardized MLOps pipelines** by integrating MLflow for experiment tracking and architected a star-schema foundation for a Power BI executive risk dashboard.

### Option 2: Machine Learning & Forecasting Focus (If applying for more ML-heavy teams)
* **Trained a LightGBM demand forecasting model** on hierarchical sales data, engineering 28-day lagged and rolling features to achieve a 0.4701 WAPE, outperforming seasonal moving average baselines.
* **Implemented Quantile Regression (P90) models** to accurately size dynamic safety stock and calculate target Reorder Points (ROP), successfully isolating intermittent demand signals.
* **Orchestrated model evaluation using MLflow**, implementing rigorous time-series holdout validation via Pytest to guarantee zero temporal data leakage during feature generation.

### Option 3: Operations Research & Optimization Focus (If applying for pure Operations/Placement teams)
* **Designed a network inventory allocation solver** using Python and PuLP to route centralized stock to regional nodes under strict warehouse capacity constraints.
* **Formulated a cost-minimization objective function** balancing interstate transfer penalties (\$0.50/unit) against high-risk stockout penalties (\$2.00/unit) for optimal placement.
* **Achieved a 7.1% cost reduction** compared to naive proportional allocation strategies by strategically withholding and routing inventory to high-margin fulfilment centers.

### Interview Talking Points for HP
1. **"I know how to prevent Data Leakage"**: Be ready to talk about why your `create_lags()` function shifts by 28 days and why time-series cross-validation is non-negotiable.
2. **"I understand Intermittent Demand"**: Mention how you used Syntetos-Boylan categorization in EDA to prove that sparse sales require quantile regression rather than just predicting the mean.
3. **"I built this for reality, not Kaggle"**: Emphasize that you didn't just build a model; you translated the ML forecast into a concrete operations decision (Reorder Point) and fed it into an LP solver to handle real-world constraints (capacities and transfer costs).
4. **"I didn't fake a Big Data environment"**: Admit that while you ran it locally on Pandas/Parquet due to RAM limits, you specifically designed the pipeline as a Bronze/Silver/Gold Medallion architecture so it would perfectly map to HP's enterprise Databricks environment.
