# Phase 1: data schema, generators, baseline

Rebuild everything with `python -m src.build_all` (seed 42, ~5 s). Tests: `python -m pytest -q`.

## Why synthetic (and M5-shaped)
Fully synthetic, with the same grain as M5/Walmart (weekly store x SKU x sell_price x units). M5 has no margins, inventory,
competitor prices, segments or ground-truth elasticities, so it would need synthetic data for ~everything anyway.
With synthetic data we *know* the true elasticities, so we can prove the estimator and test the planner against a known
"real world". The estimation code only needs a table shaped like `sales_history.csv`, so real M5 can be swapped in.

## Tables (data/)
| file | grain | key columns |
|---|---|---|
| products | SKU (48, 6 categories) | sku, name, category, subcategory, brand, is_private_label, list_price, unit_cost, shelf_life_days |
| stores | store (8, 4 regions) | store_id, city, region, sq_ft |
| sales_history | week x store x SKU (104 weeks, 39,936 rows) | week_start, store_id, sku, sell_price, units, on_promo |
| basket_items | basket x SKU (30k baskets) | basket_id, store_id, sku (for affinity / relationship agent) |
| inventory | store x SKU, projected at window start | on_hand_units, weekly_resupply_units, resupply_start_date, days_to_expiry |
| competitor_prices | competitor x SKU, snapshot 2026-10-02 (stale on purpose) | competitor, sku, price, on_promo |
| holidays, holiday_lifts | holiday dates; category lift | holiday_date, name / name, category, lift |
| segments, segment_category, store_segment_mix | customer segments | customer_share; unit_share + elasticity_multiplier per category; segment mix per store |

## Messy text inputs (data/inputs/) - the Phase 3 Input Agent reads these
`brand_guidelines_email.txt`, `competitor_price_note.txt`, `budget_memo.txt`, `ops_inventory_note.txt`.
Deliberate traps (see `gold/extraction_traps.json`): superseded budget, contingent $5k, unverified rumour, a competitor promo that
expires before the window, informal product names, no audience guidance at all.

## Ground truth (data/gold/) - never shown to agents
`true_demand_params.csv` (base units, own elasticity, seasonality), `true_cross_elasticity.csv` (48x48, substitutes > 0,
complements < 0), `region_category_index.csv`, `constraints.json` (what the Input Agent should extract), `extraction_traps.json`.

## Demand model (src/demand.py)
`units = base x store_size x region x season x holiday x exp( E*ln(r) + C @ ln(r) )`, r = price / list price, Poisson noise.
Own-elasticity E and cross-elasticity C (cannibalization / halo). History has random chain-wide promos so both are identifiable.

## Evaluator and validator (src/evaluator.py)
`evaluate_plan(plan)` runs a plan through the true demand model with inventory dynamics (finite stock, resupply capped at the normal
run rate, supply outage, expiry write-off) and returns KPIs. `check_constraints(...)` is the deterministic validator
(budget, blended promo margin, per-SKU margin floor, discount caps, no-promo SKUs, consecutive weeks, per-category cap,
mutual exclusion, bundles, regional blackouts, clearance targets). Plan format: `week_start, store_id, sku, discount_pct`.

## Baseline result (flat 20% off everything, 5 weeks, all stores)
See `data/baseline_results.json` after a build. Headline: revenue +7% vs no promo, gross margin 31.3% -> 14.2%, spend about 3.3x the
budget, 11 constraint rules broken, sunscreen clearance target still missed.
