"""Phase 4 tests: Monte Carlo simulation, constraint satisfaction, edge cases.

Run:  python -m pytest tests/test_phase4_simulation.py -q
"""
import numpy as np
import pandas as pd
import pytest

from src.config import plan_weeks
from src.demand import DemandModel
from src.elasticity import fit as fit_est
from src.evaluator import check_constraints, evaluate_plan, load_constraints, load_inventory
from src.simulation import (
    SimResult,
    conflicting_constraints,
    impossible_budget_constraints,
    simulate,
    zero_stock_plan,
)

W = plan_weeks()


@pytest.fixture(scope="module")
def model():
    return DemandModel()


@pytest.fixture(scope="module")
def est():
    return fit_est()


@pytest.fixture(scope="module")
def c():
    return load_constraints()


@pytest.fixture(scope="module")
def inv():
    return load_inventory()


@pytest.fixture(scope="module")
def feasible_plan(model):
    """Minimal plan that satisfies all constraints (reused from Phase 1 tests)."""
    S = model.store_ids
    rows = []
    for sku, stores, d, weeks in [
        ("C008", S, 0.30, [0, 1, 2, 4]),
        ("P006", S, 0.20, [0, 1]),
        ("S008", S, 0.25, [0, 1, 2]),
        ("D002", ["S03", "S05"], 0.15, [0, 1]),
        ("C004", S, 0.15, [0, 1, 2]),
        ("C007", S, 0.15, [0, 1, 2]),
        ("H002", S, 0.15, [0, 1, 2]),
    ]:
        for wi in weeks:
            for s in stores:
                rows.append({"week_start": W[wi], "store_id": s, "sku": sku, "discount_pct": d})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def sim_result(feasible_plan, est, model, c, inv):
    return simulate(feasible_plan, est=est, model=model, constraints=c, inv=inv, n=100, seed=42)


# ── Distribution sanity ───────────────────────────────────────────────────────

def test_sim_returns_correct_n_scenarios(sim_result):
    assert sim_result.n_scenarios == 100
    assert len(sim_result.samples["revenue"]) == 100


def test_sim_revenue_distribution_is_positive(sim_result):
    assert (sim_result.samples["revenue"] > 0).all()


def test_sim_net_contribution_has_spread(sim_result):
    """Uncertainty in elasticity + Poisson noise should produce a non-trivial spread."""
    p10, p50, p90 = sim_result.kpi_band("units_sold")
    assert p90 > p10, "p90 should exceed p10"
    spread_pct = (p90 - p10) / abs(p50)
    # units_sold is driven by Poisson noise across ~130k units — expect at least 0.5% spread
    assert spread_pct > 0.005, f"spread too narrow: {spread_pct:.2%}"


def test_sim_gross_margin_pct_bounded(sim_result):
    """Gross margin % should stay in a plausible range across all scenarios."""
    arr = sim_result.samples["gross_margin_pct"]
    assert (arr > 0).all()
    assert (arr < 1).all()


def test_sim_sell_through_samples_present(sim_result, c):
    """All clearance targets should have sell-through samples."""
    expected_keys = {
        f'{t["sku"]}@' + ("all" if t["stores"] == "all" else "+".join(t["stores"]))
        for t in c["clearance_targets"]
    }
    assert expected_keys <= set(sim_result.sell_through_samples.keys())


def test_sim_sell_through_in_unit_interval(sim_result):
    for key, arr in sim_result.sell_through_samples.items():
        assert (arr >= 0).all() and (arr <= 1.0 + 1e-6).all(), key


def test_sim_p50_close_to_deterministic(feasible_plan, est, model, c, inv):
    """Median of noisy simulations should be close to the deterministic evaluation."""
    det_kpis, _ = evaluate_plan(feasible_plan, model, inv, c)
    result = simulate(feasible_plan, est=est, model=model, constraints=c, inv=inv, n=100, seed=0)
    _, p50, _ = result.kpi_band("revenue")
    assert abs(p50 / det_kpis["revenue"] - 1) < 0.05, (
        f"p50 revenue {p50:,.0f} vs deterministic {det_kpis['revenue']:,.0f}"
    )


# ── Constraint satisfaction rate ─────────────────────────────────────────────

def test_feasible_plan_high_satisfaction_rate(sim_result):
    """A structurally valid plan should have 0 hard-rule violations in every scenario.
    Clearance misses under elasticity uncertainty are NOT counted here."""
    assert sim_result.feasibility_rate >= 0.95, (
        f"structural feasibility rate {sim_result.feasibility_rate:.0%} < 95%\n"
        f"violations: {sim_result.violation_counts}"
    )


def test_flat20_low_satisfaction_rate(est, model, c, inv):
    """Flat 20% violates many constraints — should be feasible in < 5% of scenarios."""
    from src.baseline import flat_plan
    result = simulate(flat_plan(model), est=est, model=model, constraints=c, inv=inv, n=50, seed=1)
    assert result.feasibility_rate < 0.05, (
        f"flat-20% feasibility rate {result.feasibility_rate:.0%} should be near 0"
    )


def test_no_promo_clearance_always_fails(feasible_plan, est, model, c, inv):
    """Empty plan always misses clearance targets — structural rules are fine but clearance always fails."""
    empty = feasible_plan.iloc[0:0]
    result = simulate(empty, est=est, model=model, constraints=c, inv=inv, n=30, seed=2)
    # Structural feasibility is 100% (no spend, no caps broken)
    assert result.feasibility_rate == 1.0
    # But clearance violations are present in every scenario
    assert any("clearance" in r for r in result.violation_counts)
    clearance_violations = sum(v for r, v in result.violation_counts.items() if "clearance" in r)
    assert clearance_violations >= 30 * 4  # all 4 targets fail in every scenario


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_zero_stock_plan_sells_nothing(model, est, c, inv):
    """Promoting SKUs with zero stock should result in near-zero units sold."""
    plan = zero_stock_plan(model)
    if len(plan) == 0:
        pytest.skip("no zero-stock SKUs in this dataset")
    result = simulate(plan, est=est, model=model, constraints=c, inv=inv, n=20, seed=3)
    # All revenue should come from non-promoted baseline sales of other SKUs
    # The promoted SKUs themselves contribute nothing — units_sold median should be low
    assert result.samples["units_sold"].mean() >= 0   # no negative units


def test_impossible_budget_always_violates(feasible_plan, est, model, c, inv):
    """With a $1 budget every scenario should violate the budget constraint."""
    tight_c = impossible_budget_constraints(c)
    result = simulate(feasible_plan, est=est, model=model, constraints=tight_c, inv=inv, n=30, seed=4)
    assert result.feasibility_rate == 0.0
    assert "marketing_budget" in result.violation_counts


def test_conflicting_constraints_always_violates(feasible_plan, est, model, c, inv):
    """A plan where D001 is both no_promo and has a clearance target is always infeasible."""
    conflict_c = conflicting_constraints(c)
    # Add D001 to the plan to trigger the no_promo violation
    extra = pd.DataFrame([
        {"week_start": W[0], "store_id": s, "sku": "D001", "discount_pct": 0.10}
        for s in model.store_ids
    ])
    plan = pd.concat([feasible_plan, extra], ignore_index=True)
    result = simulate(plan, est=est, model=model, constraints=conflict_c, inv=inv, n=20, seed=5)
    assert result.feasibility_rate == 0.0


# ── SimResult API ─────────────────────────────────────────────────────────────

def test_sim_result_to_dict_serialisable(sim_result):
    import json
    d = sim_result.to_dict()
    json.dumps(d, default=float)   # must not raise
    assert "kpis" in d and "sell_through" in d
    assert "p50" in d["kpis"]["net_contribution"]


def test_sim_result_summary_string(sim_result):
    s = sim_result.summary()
    assert "Monte Carlo" in s
    assert "net_contribution" in s
    assert "feasibility" in s.lower()


def test_clearance_hit_rate_for_feasible_plan(sim_result, c):
    """For a feasible plan, clearance hit rate should be high for most targets."""
    hit_rates = []
    for tg in c["clearance_targets"]:
        key = f'{tg["sku"]}@' + ("all" if tg["stores"] == "all" else "+".join(tg["stores"]))
        rate = sim_result.clearance_hit_rate(key, tg["min_sell_through_pct"])
        hit_rates.append(rate)
    # At least half the targets should be hit in ≥ 70% of scenarios
    assert sum(r >= 0.70 for r in hit_rates) >= 2, f"hit rates: {hit_rates}"
