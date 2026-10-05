# FlowCast: Demand Forecasting and Inventory Placement Optimizer

*(Note: The M5 Forecasting dataset from Kaggle is used here as a stand-in for a supply chain structure. The states CA, TX, WI are treated as regional hubs, and the 10 stores are treated as fulfilment nodes. This is NOT actual HP data.)*

An end-to-end Machine Learning and Operations Research prototype designed for the HP "AI for Placement and Inventory Management" Data Science Intern role. 

This project tackles two core supply chain challenges:
1. **Demand Forecasting**: Predicting SKU-level daily demand using LightGBM (Point P50 and Quantile P90 models) with strict temporal holdout validation.
2. **Inventory Placement**: Optimizing the allocation of inventory across a constrained fulfilment network to minimize transfer, holding, and stockout costs using Linear Programming (PuLP).

## 🚀 Key Results
- **Forecasting Excellence**: LightGBM outperformed the 28-day Moving Average baseline on WAPE (0.4701 vs 0.4737) and achieved near-perfect prediction interval coverage (90.6% on a 90% target).
- **Network Optimization**: The PuLP Linear Program saved **$1,520 (7.15%)** per week compared to a naive proportional allocation by strategically balancing $2.00 stockout costs against $0.50 interstate transfer costs under strict warehouse capacity limits.
- **Robust Engineering**: Implemented Medallion architecture (Bronze/Silver/Gold) using Pandas/Parquet (simulating PySpark/Delta), strict rolling-origin cross-validation, MLflow experiment tracking, and mathematical unit tests.

## 🛠 Tech Stack
- **Data Engineering**: Pandas, PyArrow (Parquet) simulating Databricks Bronze/Silver/Gold pipelines.
- **Machine Learning**: LightGBM (Gradient Boosting, Quantile Regression).
- **MLOps**: MLflow (SQLite backend) for parameter, metric, and model registry tracking.
- **Operations Research**: PuLP (Linear Programming) with the GLPK solver.
- **BI / Visualization**: Power BI (Specs provided below) and Matplotlib.
- **Testing**: Pytest (strict leakage checks, mathematical unit tests).

## 📂 Project Structure
- `configs/params.yaml`: Centralized configuration and synthetic assumptions (lead times, costs, capacities).
- `src/`: Core modules (bronze, silver, baselines, features, metrics, model, quantile, inventory, optimizer).
- `tests/`: Pytest suite verifying logic and preventing data leakage.
- `data/`: Local Medallion architecture storage (Bronze, Silver, Gold).
- `ASSUMPTIONS.md`: Running log of all verified findings and constraints.

---

## Phase 5: PuLP Placement Optimizer Mathematical Formulation

Before diving into the code, here is the exact Linear Programming formulation used to optimize inventory allocation across the 10 stores.

### Indices
* $i \in I$: Items
* $s \in S$: Stores (Fulfilment Nodes)

### Parameters (from `configs/params.yaml`)
* $R_{i,s}$: Reorder Point (target inventory demand) for item $i$ at store $s$ over the next 7 days.
* $B_i$: Total inventory batch available for item $i$ (we assume we produce exactly enough to meet total network demand: $B_i = \sum_{s} R_{i,s}$).
* $C_s$: Maximum storage capacity of store $s$ in units.
* $T_s$: Transfer cost to store $s$ per unit (\$0.10 for CA nodes, \$0.50 for TX/WI nodes, assuming origin in CA).
* $H$: Holding cost per unit of excess inventory (\$0.50).
* $K$: Stockout cost per unit of unmet ROP (\$2.00).
* $M_{i,s}$: Minimum allocation requirement ($0.05 \times R_{i,s}$).

### Decision Variables
* $X_{i,s} \ge 0$: Units of item $i$ allocated to store $s$.
* $U_{i,s} \ge 0$: Under-allocation (Stockout units) for item $i$ at store $s$.
* $O_{i,s} \ge 0$: Over-allocation (Excess holding units) for item $i$ at store $s$.

### Objective Function
Minimize Total Expected Cost (Transfer + Stockout + Holding):
$$ \text{Minimize } Z = \sum_{i \in I} \sum_{s \in S} \left( T_s X_{i,s} + K U_{i,s} + H O_{i,s} \right) $$

### Constraints

1. **Total Inventory Conservation:** We must allocate the entire available batch for each item.
   $$ \sum_{s \in S} X_{i,s} = B_i \quad \forall i \in I $$

2. **Store Capacity:** A store cannot receive more units across all items than its maximum capacity.
   $$ \sum_{i \in I} X_{i,s} \le C_s \quad \forall s \in S $$

3. **Minimum Allocation:** Ensure every store gets at least a base fraction of its demand (prevents fully starving a region).
   $$ X_{i,s} \ge M_{i,s} \quad \forall i \in I, \forall s \in S $$

4. **Under/Over Allocation Balancing:** Defines the $U_{i,s}$ and $O_{i,s}$ variables relative to the target Reorder Point $R_{i,s}$.
   $$ X_{i,s} + U_{i,s} - O_{i,s} = R_{i,s} \quad \forall i \in I, \forall s \in S $$

## Power BI Dashboard Specifications

The outputs of this project in `data/gold/` are explicitly structured as a star-schema foundation for a Power BI dashboard.

### 1. Data Sources (Gold Layer)
* **Fact_Inventory_Forecast** (`data/gold/inventory_forecast.parquet`): Daily grain. Contains actual sales, P50 point forecast, P90 quantile forecast, Lead Time Demand, Safety Stock, Reorder Point, and boolean flags for Stockouts/Overstocks.
* **Fact_Allocations** (`data/gold/allocations/optimal.parquet`): Snapshot grain. Contains the final optimizer allocations per item per store vs the ROP target.

### 2. Suggested Dashboard Layout

**Page 1: Forecast & Risk Overview (Executive View)**
* **KPI Cards**: Overall WAPE, Total Expected Stockouts, Total Expected Overstocks, Average Safety Stock units.
* **Line Chart (Time Series)**: `date` on X-axis. `sales` (actual) vs `lgbm_forecast` (predicted). Add a shaded error band using `p90_forecast`.
* **Matrix/Table**: Store (rows) by Category (columns) showing `stockout_flag` counts conditionally formatted (Red for high).

**Page 2: Optimizer vs Naive Allocation (Operations View)**
* **KPI Cards**: Total Optimizer Cost, Total Naive Cost, Total Savings ($).
* **Bar Chart**: `store_id` on X-axis. Stacked bars for `optimal_alloc` vs `rop_target`. Shows which stores were starved by the capacity constraint.
* **Scatter Plot**: `optimal_alloc` vs `rop_target`. Any point below the $y=x$ line represents a strategic stockout chosen by the LP.
