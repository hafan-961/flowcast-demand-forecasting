import sys; sys.path.insert(0, '.')
import pandas as pd
from src.config import load_config
from src.optimizer import allocate_inventory
import os

cfg = load_config()

print('Loading gold inventory data ...')
gold = pd.read_parquet('data/gold/inventory_forecast.parquet')

rop_df = gold.groupby(['item_id', 'store_id', 'state_id'])['reorder_point'].mean().reset_index()
rop_df.rename(columns={'reorder_point': 'rop'}, inplace=True)
rop_df['rop'] = rop_df['rop'].fillna(0)

total_rop_by_store = rop_df.groupby('store_id')['rop'].sum()
print('\nTotal ROP Demand by Store:')
print(total_rop_by_store)

max_demand = total_rop_by_store.max()
caps = {}
for s in rop_df['store_id'].unique():
    demand = total_rop_by_store[s]
    # Set capacity to 90% of actual demand to simulate a constrained network
    caps[s] = int(demand * 0.90) if demand > 0 else 1000

print('\nApplying Capacities (90% of target to force constrained allocation):')
print(caps)

opt_df, naive_df, summary = allocate_inventory(
    rop_df=rop_df,
    capacities=caps,
    transfer_costs=cfg['optimizer']['transfer_cost_per_unit'],
    holding_cost=cfg['inventory']['holding_cost_per_unit'],
    stockout_cost=cfg['inventory']['stockout_cost_per_unit'],
    min_alloc_pct=cfg['optimizer']['min_allocation_pct']
)

print('\n--- OPTIMIZATION RESULTS ---')
for k, v in summary.items():
    if 'cost' in k or 'savings' in k:
        print(f'{k}: ${v:,.2f}')
    else:
        print(f'{k}: {v}')
        
# Save optimizer results
os.makedirs('data/gold/allocations', exist_ok=True)
opt_df.to_parquet('data/gold/allocations/optimal.parquet', index=False)
naive_df.to_parquet('data/gold/allocations/naive.parquet', index=False)
print('\nSaved allocations to data/gold/allocations/')
