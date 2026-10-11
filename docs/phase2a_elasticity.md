# Phase 2a: elasticity estimation (written 2026-10-11, NOT yet run)

Status: code and tests written without being executed. Run the commands below locally and tick the Phase 2 boxes in CHECKPOINTS.md once they pass.

## Run
```
python -m src.build_all                                   # Phase 1 data (verify first)
python -m pytest -q                                       # expect the 20 Phase 1 tests to pass
python -m src.elasticity                                  # writes data/estimates/*.csv and prints a summary
python -m pytest tests/test_phase2_elasticity.py -q       # 7 new tests
```

## What was added
| file | purpose |
|---|---|
| `src/elasticity.py` | Estimates own and cross price elasticities and a baseline-demand forecaster from `sales_history.csv` only |
| `tests/test_phase2_elasticity.py` | Checks the estimates against the known world |
| `data/estimates/` (created by `python -m src.elasticity`) | `own_elasticity.csv`, `cross_elasticity.csv`, `relationships.csv` |

## Method
One equation per SKU k, one row per (week, store):

`ln(units + 0.5) - ln(holiday_factor) = store effect + annual seasonality + E_k ln(r_k) + sum_{j != k} C_kj ln(r_j) + noise`, with `r = sell_price / list_price`.

- All SKUs share the same regressors, so all 48 equations are solved in one least-squares call.
- Seasonality: 4 sine/cosine pairs per SKU (enough for the sharply peaked sunscreen season). Store effects absorb store size and region.
- Holiday lifts are taken as given from `holidays.csv` and `holiday_lifts.csv` (used as an offset, not estimated).
- `ln(units + 0.5)` is nearly unbiased for Poisson counts, so low-volume SKUs are not biased.
- Identification: promos are random and independent across SKUs and weeks, so own and cross effects are separable. Rough standard error is about 0.07 to 0.10 per coefficient (back-of-envelope, to be confirmed by the run).
- Orientation of C: row = SKU whose demand responds, column = SKU whose price moves. Substitutes are positive, complements negative. This matches the ground-truth matrix.
- Raw OLS gives about 2,200 cross coefficients, most of them noise. A significance screen keeps a link if |t| >= 2.0 for a positive link within the same subcategory, or |t| >= 3.5 otherwise (`T_SAME_SUBCATEGORY`, `T_OTHER`).
- The estimator never imports `src/demand.py` or reads the gold folder; a test enforces this.

## API for the optimizer and Monte Carlo
```python
from src.elasticity import fit
est = fit()
est.E, est.E_se            # own elasticity and standard error, shape (48,)
est.C, est.C_raw, est.C_se # screened / raw / standard error, shape (48, 48)
est.baseline_units(ws)     # (store, sku) units at regular price for the week starting ws
est.expected_units(ws, r)  # same, with a price-ratio matrix r
est.draw(rng)              # one plausible parameter set (Phase 4 Monte Carlo)
est.pairs()                # tidy substitute / complement table (for the Phase 3 Relationship agent)
```

## Test thresholds (deliberately loose; tighten after the first run)
- Own elasticity: all negative, mean abs error < 0.15, max < 0.40, at least 85% of true values within 3 standard errors.
- Cross elasticity, for true links with |C| >= 0.25: sign agreement >= 95%, mean abs error < 0.15, at least 80% kept by the screen, at most 8 invented links.
- Baseline forecast for the planning weeks, per SKU: max error < 20%, mean error < 5%.
- A 20% discount on B001 gives its own lift within 10% of the true model, and cannibalises B002.

## Phase 2 objective: recommendation (pending your OK)
- "Do nothing" misses all four clearance targets (sunscreen 29% vs 55%, ice cream 59% vs 70%, trail mix 51% vs 65%, yogurt 63% vs 80%), so it is infeasible. That is the answer to "why not do nothing".
- Headline objective: maximise net contribution subject to every hard constraint (least-cost feasible plan), with the revenue weight lambda defaulting to 0.
- Expose lambda as a slider / frontier in the demo. The only data-backed reason for a revenue weight is the budget memo saying Thanksgiving week matters most for the board numbers; if used, weight that week.
- Report results against both baselines (flat 20% and no promo).

## Correction made in CHECKPOINTS.md
The note compared do-nothing gross margin ($194.5k) with flat-20% net contribution ($65k). Like for like: gross margin $194.5k vs $94.5k, or net contribution $193.9k vs $65.5k.

## Things the optimizer will hit (design notes for the next step)
- **Blended margin guardrail (21%)** applies to promoted sales only. A 35% sunscreen promo earns about 15%, so the plan will add high-margin store-brand promos to lift the blend (the Phase 1 feasible test plan already does this). Be ready to explain it.
- **Sunscreen clearance is borderline:** 29% to 55% sell-through under a 35% cap and a 3-week run limit. Plan against a buffer or a pessimistic elasticity (E plus a multiple of its standard error), otherwise the true-model evaluator may show a miss.
- **Audience:** the plan format has no segment dimension and the demand model has no segment effect. Treat audience as store / region targeting and do not claim segment targeting.
- **Competitors:** the demand model has no competitor term, so competitor awareness cannot be verified by the evaluator except through a scenario shock.
- **Budget will likely not bind:** a minimal clearance plan is roughly $10k of the $59k (back-of-envelope, unverified).
- **Salsa halo:** by hand calculation, a salsa discount adds about $4 to $5 gross margin per store-week through the chips halo, but the $15 execution cost removes it. Let the optimizer show this rather than assert it.

## Next step
Build `src/optimizer.py` (discrete discount grid per store, week and SKU, solved with PuLP, constraints from the constraints dict, cross effects within linked SKU groups), then check its plan with `evaluate_plan` and `check_constraints`.
