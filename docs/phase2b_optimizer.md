# Phase 2b: Optimizer (written 2026-10-11)

Status: **verified locally: CBC Optimal, zero evaluator violations (2026-10-11)**.

## What was built
| File | Purpose |
|---|---|
| `src/optimizer.py` | PuLP MIP optimizer: binary x[w,s,k,d] variables, all hard constraints from constraints.json, objective = net_contribution + λ × revenue_lift |
| `data/optimizer_results.json` | Solver status, validated KPIs, planning slack and evaluated clearance shortfalls |
| `data/optimizer_plan.csv` | Feasible integer promo plan; failed solves are never rounded into plans |

## Run
```
python -m src.optimizer
```

## Method
- Decision variable: `x[w, s, k, d] ∈ {0,1}` — promote SKU k in store s in week w at discount depth d
- Discount grid: 10%, 15%, 20%, 25%, 30%, 35%
- Demand uplift pre-computed from `DemandEstimates.expected_units` (own-elasticity only; cross effects verified post-solve via `evaluate_plan`)
- All hard constraints encoded: no_promo_skus, max_discount (global / brand / private-label), min_sku_margin, blackouts, mutually_exclusive, promote_together, max_consecutive_weeks, max_skus_per_category_per_store_week, marketing_budget, min_blended_promo_margin
- Clearance targets are soft: nonnegative unit slack allows a shortfall, penalized at $1,000 per unit in the objective. Expiry-aware week blocking and the per-SKU discount floor remain enforced.
- Planning slack measures the buffered demand approximation; evaluated sell-through shortfalls are reported separately in percentage points. Neither is a hard-rule violation.
- Results are written only after an integer solution exists and the full-demand evaluator finds no hard-rule violations. A failed solve raises an error instead of publishing an arbitrary rounded point.
- Solver: CBC via PuLP, 120s time limit
- Live pipeline calls use a 15s limit per attempt; `candidate_skus` restricts selection to the LLM shortlist plus clearance and pairing dependencies. Standalone defaults remain unchanged.

## Verified Results (Solver Status: Optimal)
| Scenario | Revenue | GM% | Net Contrib | Spend |
|---|---|---|---|---|
| No Promo | $621,325 | 31.3% | $193,919 | $0 |
| Flat 20% | $665,024 | 14.2% | $65,456 | $195,056 |
| Optimizer | $639,103 | 27.1% | $161,803 | $51,699 |

Evaluated sell-through: C008 56.25% (target 55%), P006 73.14% (70%), S008 66.32% (65%), D002@S03+S05 80.56% (80%). All four targets are met in this deterministic evaluation.

D002 has 31.51 units of planning slack against the buffered model, despite meeting the target in the full-demand evaluation. All evaluated percentage-point shortfalls are zero.

Against flat-20%, net contribution improves by $96,347 and spend falls by $143,357. Revenue and units sold are lower, and expired units are higher: this is not an improvement on every metric. No-promo still has higher contribution but misses all four clearance targets.

## Reliability Boundaries

The previous infeasible solve's rounded values were not a solution or a best bound. Those artifacts have been replaced. Soft clearance avoids infeasibility from unattainable targets while keeping margin caps, budget, blackouts, pairing and consecutive-week rules hard.

`Optimal` refers to the MIP approximation, not global optimality in the full nonlinear demand model. Demand uncertainty can still cause target misses or budget/margin failures; simulation reports those risks separately. Contradictory hard constraints or a solver failure can still prevent a solution and must not be presented as a valid plan.

Regression command: `python -m pytest tests/test_phase2_optimizer.py -q` (5 passed), including zero budget, prohibited clearance SKU, and failed-solve rejection.

## Decisions confirmed
- Objective: maximise the existing contribution proxy + 0.10 × revenue_lift minus the clearance penalty (λ=0.10).
- Report against both baselines (no-promo and flat-20%)
- "Do nothing" is hard-rule feasible but misses all four soft clearance targets: C008 sell-through is ~29% vs 55%.

## Next step
Verify the UI and simulation against the regenerated plan before recording the submission demo.
