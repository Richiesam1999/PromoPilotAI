# Phase 4: Simulation and Reliability (written 2026-10-11)

Status: **code complete, tests passing (15 passed, 1 skipped)**.

## Run
```
python -m src.simulation                              # full 3-scenario run, writes data/simulation_results.json
python -m pytest tests/test_phase4_simulation.py -q  # 15 passed, 1 skipped (no LLM, deterministic)
```

## What was added
| File | Purpose |
|---|---|
| `src/simulation.py` | Monte Carlo engine: N draws of `est.draw(rng)` + Poisson noise → KPI distributions + structural feasibility rate |
| `tests/test_phase4_simulation.py` | 16 tests: distribution sanity, constraint satisfaction rate, edge cases |

## Method

Each scenario:
1. **Perturb elasticity** — `est.draw(rng)` samples E and C from their estimated distributions (Phase 2a standard errors). This captures model uncertainty: we don't know the true elasticities exactly.
2. **Swap into the demand model** — the drawn E and C replace the model's parameters for this scenario only (restored in a `finally` block).
3. **Poisson demand noise** — `evaluate_plan(..., noise=True)` draws actual units from `Poisson(mean)` instead of using the mean directly. This captures execution uncertainty.
4. **Evaluate KPIs** — full inventory dynamics (stock depletion, resupply, expiry write-off).
5. **Check constraints** — `check_constraints(...)` records all violations. Violations are split into two categories:
   - **Structural** (budget, margin caps, no-promo, consecutive weeks, blackouts, etc.) — these are deterministic given the plan structure and count against `feasibility_rate`.
   - **Clearance** — sell-through targets can fail under weaker-than-expected elasticity draws even for a structurally valid plan. These are tracked separately via `sell_through_samples` and `clearance_hit_rate()`, not against `feasibility_rate`.

```
for i in range(N):
    drawn_est  = est.draw(rng)          # perturb E, C by their standard errors
    model.E, model.C = drawn_est.E, drawn_est.C
    kpis, _    = evaluate_plan(plan, model, inv, c, noise=True, rng=rng)
    violations = check_constraints(plan, kpis, c, ...)
    record(kpis, violations)
```

Default N = 200. Runtime: ~10–15 s for 200 scenarios on a laptop.

## API

```python
from src.simulation import simulate, simulate_scenarios, SimResult

# Single plan
result = simulate(plan, est=est, model=model, constraints=c, inv=inv, n=200, seed=0)

# Multiple plans compared
results = simulate_scenarios(
    {"optimizer": opt_plan, "flat_20pct": flat_plan, "no_promo": empty_plan},
    est=est, model=model, constraints=c, inv=inv, n=200,
)

# Accessors
result.kpi_band("net_contribution")          # (p10, p50, p90)
result.sell_through_band("C008@all")         # (p10, p50, p90)
result.clearance_hit_rate("C008@all", 0.55)  # fraction of scenarios hitting target
result.feasibility_rate                      # fraction with zero STRUCTURAL violations (clearance tracked separately)
result.violation_counts                      # {"marketing_budget": 12, ...}
result.summary()                             # formatted percentile table (string)
result.to_dict()                             # JSON-serialisable percentile summary
```

## Output: `data/simulation_results.json`

Written by `python -m src.simulation`. Structure:
```json
{
  "optimizer": {
    "plan_label": "optimizer",
    "n_scenarios": 200,
    "feasibility_rate": 0.97,
    "violation_counts": {},
    "kpis": {
      "net_contribution": {"p5": ..., "p10": ..., "p25": ..., "p50": ..., "p75": ..., "p90": ..., "p95": ...},
      ...
    },
    "sell_through": {
      "C008@all": {"p5": ..., "p50": ..., "p90": ...},
      ...
    }
  },
  "flat_20pct": { ... },
  "no_promo":   { ... }
}
```

## Expected results (from known world)

| Scenario | Structural feasibility rate | Net contrib p50 | Clearance violations |
|---|---|---|---|
| Optimizer plan | ≥ 95% | ~$160k | low (plan is built to clear) |
| Flat 20% | < 5% | ~$65k | present (budget, margin, caps all broken) |
| No promo | 100% | ~$193k | always — all 4 targets fail in every scenario |

Key insight for the demo: the optimizer plan is **structurally reliable** across uncertainty. No-promo is structurally clean but always fails clearance. Flat-20% fails on hard rules in almost every scenario.

## Edge cases tested

| Scenario | Expected behaviour |
|---|---|
| Zero-stock SKUs promoted | Units sold = 0 for those SKUs; no negative revenue |
| Budget = $1 (impossible) | 100% of scenarios violate `marketing_budget` |
| D001 in both `no_promo_skus` and `clearance_targets` | 100% of scenarios violate `no_promo_sku` |
| Empty plan (no promo) | Structural feasibility = 100% (no spend); clearance violations in every scenario |

## Confidence flags surfaced (Phase 3 → Phase 4 link)

The simulation result bands are the quantitative backing for the confidence flags the Input Agent emits:
- If `clearance_hit_rate("C008@all", 0.55) < 0.80` → surface a low-confidence warning in the UI
- If `feasibility_rate < 0.90` → flag the plan as "needs review before launch" (structural hard rules at risk)
- Clearance hit rate is reported separately per target so the UI can show which SKUs are at risk under uncertainty

## Test coverage (`tests/test_phase4_simulation.py`)

| Test | What it checks |
|---|---|
| `test_sim_returns_correct_n_scenarios` | Array length = N |
| `test_sim_revenue_distribution_is_positive` | All revenue samples > 0 |
| `test_sim_net_contribution_has_spread` | units_sold p90 > p10, spread > 0.5% (Poisson noise across ~130k units) |
| `test_sim_gross_margin_pct_bounded` | All samples in (0, 1) |
| `test_sim_sell_through_samples_present` | All 4 clearance targets have samples |
| `test_sim_sell_through_in_unit_interval` | All sell-through in [0, 1] |
| `test_sim_p50_close_to_deterministic` | Median within 5% of deterministic eval |
| `test_feasible_plan_high_satisfaction_rate` | Structurally valid plan ≥ 95% structural feasibility rate |
| `test_flat20_low_satisfaction_rate` | Flat-20% < 5% structural feasibility rate (budget + margin caps broken) |
| `test_no_promo_clearance_always_fails` | Empty plan: structural feasibility = 100%, clearance violations in every scenario (≥ 120 total) |
| `test_zero_stock_plan_sells_nothing` | No negative units |
| `test_impossible_budget_always_violates` | 100% budget violation |
| `test_conflicting_constraints_always_violates` | 100% violation when D001 is both no_promo and clearance target |
| `test_sim_result_to_dict_serialisable` | `to_dict()` is JSON-serialisable, has p50 |
| `test_sim_result_summary_string` | `summary()` contains expected headings |
| `test_clearance_hit_rate_for_feasible_plan` | ≥ 2 of 4 targets hit in ≥ 70% of scenarios |

## Next step
Phase 5: UI — Streamlit app showing extracted parameters, plan comparison, simulation bands, live re-plan demo, validator rejection.
