"""
PuLP Linear Programming optimizer for inventory allocation.

Solves the transportation/allocation problem to minimize costs while
respecting store capacity constraints.
"""
import pandas as pd
import pulp
from typing import Dict, Tuple

def allocate_inventory(
    rop_df: pd.DataFrame, 
    capacities: dict, 
    transfer_costs: dict, 
    holding_cost: float, 
    stockout_cost: float,
    min_alloc_pct: float
) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Run the PuLP LP to allocate stock, and compare against a naive baseline.
    
    Parameters
    ----------
    rop_df : pd.DataFrame
        DataFrame with 'item_id', 'store_id', 'state_id', 'rop' (Reorder Point target).
    capacities : dict
        Max capacity per store_id.
    transfer_costs : dict
        'intra_state' (for CA) and 'inter_state' (for TX, WI) costs.
    holding_cost : float
        Cost per unit of over-allocation.
    stockout_cost : float
        Cost per unit of under-allocation.
    min_alloc_pct : float
        Minimum percentage of ROP a store must receive.
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, dict]
        optimal_alloc_df, naive_alloc_df, summary_metrics
    """
    items = rop_df['item_id'].unique().tolist()
    stores = rop_df['store_id'].unique().tolist()
    
    # Dictionaries for quick lookup
    R = rop_df.set_index(['item_id', 'store_id'])['rop'].to_dict()
    
    # Calculate Total Batch available per item (assume we produce exactly total network ROP)
    B = {i: sum(R.get((i, s), 0) for s in stores) for i in items}
    
    # Transfer costs per store
    T = {}
    for s in stores:
        if s.startswith('CA'):
            T[s] = transfer_costs['intra_state']
        else:
            T[s] = transfer_costs['inter_state']
            
    # Initialize the LP Problem
    prob = pulp.LpProblem("Inventory_Allocation_Optimizer", pulp.LpMinimize)
    
    # Decision Variables
    # X[i,s]: amount allocated
    X = {(i, s): prob.add_variable(f"Alloc_{i}_{s}", lowBound=0, cat='Continuous') for i in items for s in stores}
    # U[i,s]: under-allocation
    U = {(i, s): prob.add_variable(f"Under_{i}_{s}", lowBound=0, cat='Continuous') for i in items for s in stores}
    # O[i,s]: over-allocation
    O = {(i, s): prob.add_variable(f"Over_{i}_{s}", lowBound=0, cat='Continuous') for i in items for s in stores}
    
    # Objective Function
    prob += pulp.lpSum(
        T[s] * X[i, s] + stockout_cost * U[i, s] + holding_cost * O[i, s]
        for i in items for s in stores
    ), "Total_Cost"
    
    # Constraints
    # 1. Total inventory conserved per item (can withhold if capacity constrained)
    for i in items:
        prob += pulp.lpSum(X[i, s] for s in stores) <= B[i], f"Batch_{i}"
        
    # 2. Store Capacity
    for s in stores:
        prob += pulp.lpSum(X[i, s] for i in items) <= capacities[s], f"Cap_{s}"
        
    # 3. Min Allocation & 4. U/O balancing
    for i in items:
        for s in stores:
            target = R.get((i, s), 0)
            # Min allocation
            prob += X[i, s] >= min_alloc_pct * target, f"MinAlloc_{i}_{s}"
            # Balance: X + U - O = R
            prob += X[i, s] + U[i, s] - O[i, s] == target, f"Balance_{i}_{s}"
            
    # Solve
    print("Solving LP with GLPK...")
    result = prob.solve(pulp.GLPK_CMD(msg=0))
    status_code = getattr(result, 'status', 1) # Fallback if structure is different
    status = "Optimal" if status_code == 1 else f"Status_Code_{status_code}"
    print(f"Solver Status: {status}")
    
    # Extract Optimal Results
    opt_results = []
    opt_cost = pulp.value(prob.objective)
    
    for i in items:
        for s in stores:
            target = R.get((i, s), 0)
            alloc = X[i, s].varValue
            opt_results.append({
                'item_id': i,
                'store_id': s,
                'rop_target': target,
                'optimal_alloc': alloc
            })
    opt_df = pd.DataFrame(opt_results)
    
    # Compute Naive Proportional Baseline
    # Naive allocates exactly proportionally to demand. 
    # Since total demand B[i] equals the total available, naive allocation exactly matches ROP!
    # Wait, what if it violates store capacity? Naive doesn't "know" about capacity constraints,
    # so we cap it at capacity (pro-rata across items at that store) and leave the rest unallocated (stockout),
    # or just penalize the naive for over-capacity. Let's strictly cap naive at capacity to be fair.
    naive_results = []
    naive_cost = 0.0
    
    for s in stores:
        store_demand = sum(R.get((i, s), 0) for i in items)
        cap = capacities[s]
        # If store demand > capacity, we scale down allocations proportionally
        scale = min(1.0, cap / store_demand) if store_demand > 0 else 1.0
        
        for i in items:
            target = R.get((i, s), 0)
            alloc = target * scale
            
            # Naive Cost calculation
            trans_c = alloc * T[s]
            under_c = max(0, target - alloc) * stockout_cost
            over_c = max(0, alloc - target) * holding_cost  # will be 0 here since alloc <= target
            
            naive_cost += (trans_c + under_c + over_c)
            
            naive_results.append({
                'item_id': i,
                'store_id': s,
                'rop_target': target,
                'naive_alloc': alloc
            })
            
    # Naive must pay holding cost for any unallocated inventory from the batch B[i]
    naive_df = pd.DataFrame(naive_results)
    for i in items:
        total_naive_alloc = naive_df[naive_df['item_id'] == i]['naive_alloc'].sum()
        unallocated = max(0, B[i] - total_naive_alloc)
        naive_cost += unallocated * holding_cost
        
    summary = {
        'status': status,
        'optimal_total_cost': opt_cost,
        'naive_total_cost': naive_cost,
        'savings': naive_cost - opt_cost,
        'savings_pct': (naive_cost - opt_cost) / naive_cost * 100 if naive_cost > 0 else 0
    }
    
    return opt_df, naive_df, summary
