"""Optimizer regressions: clearance is soft; all policy rules remain hard."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pulp
import pytest

from src.baseline import flat_plan
from src.demand import DemandModel
from src.elasticity import fit
from src.evaluator import check_constraints, evaluate_plan, load_constraints, load_inventory
from src.optimizer import optimize, run_planning


@pytest.fixture(scope="module")
def truth():
    return DemandModel()


@pytest.fixture(scope="module")
def c():
    return load_constraints()


@pytest.fixture(scope="module")
def inv():
    return load_inventory()


@pytest.fixture(scope="module")
def est():
    return fit()


@pytest.fixture(scope="module")
def headline(est, truth, c, inv):
    return optimize(constraints=c, est=est)


def test_optimized_plan_is_valid_in_the_true_world(headline, truth, c, inv):
    plan, status = headline
    assert status == "Optimal"
    assert not plan.empty
    k, _ = evaluate_plan(plan, truth, inv, c)
    assert check_constraints(plan, k, c, truth.products, truth.stores) == []
    assert k["marketing_spend"] < c["marketing_budget"]
    assert all(value >= 0 for value in plan.attrs["clearance_slack_units"].values())
    for tg in c["clearance_targets"]:
        key = f'{tg["sku"]}@' + ("all" if tg["stores"] == "all" else "+".join(tg["stores"]))
        assert k["sell_through"][key] >= tg["min_sell_through_pct"] - 1e-9, key


def test_optimized_plan_beats_both_baselines_where_it_should(headline, truth, c, inv):
    plan, _ = headline
    k_opt, _ = evaluate_plan(plan, truth, inv, c)
    empty = plan.iloc[0:0]
    k_none, _ = evaluate_plan(empty, truth, inv, c)
    k_flat, _ = evaluate_plan(flat_plan(truth), truth, inv, c)
    v_none = check_constraints(empty, k_none, c, truth.products, truth.stores)
    v_flat = check_constraints(flat_plan(truth), k_flat, c, truth.products, truth.stores)
    assert sum(v["rule"].startswith("clearance") for v in v_none) == 4
    assert len(v_flat) >= 8
    assert k_opt["net_contribution"] > k_flat["net_contribution"] + 50000
    assert k_opt["net_contribution"] < k_none["net_contribution"]         # honest: it pays to meet the constraints
    assert k_opt["expired_units"] < k_none["expired_units"]               # and the expiring yogurt is rescued


@pytest.mark.parametrize("scenario", ["zero_budget", "prohibited_clearance"])
def test_impossible_clearance_remains_soft(scenario, est, truth, c, inv):
    changed = deepcopy(c)
    if scenario == "zero_budget":
        changed["marketing_budget"] = 0
    else:
        changed["no_promo_skus"].append("C008")
    plan, status = optimize(constraints=changed, est=est)
    assert status == "Optimal"
    assert plan.attrs["clearance_slack_units"]["C008@all"] > 0
    if scenario == "zero_budget":
        assert plan.empty
    else:
        assert "C008" not in set(plan.sku)
    kpis, _ = evaluate_plan(plan, truth, inv, changed)
    violations = check_constraints(plan, kpis, changed, truth.products, truth.stores)
    assert violations
    assert all(violation["rule"].startswith("clearance:") for violation in violations)


def test_failed_solver_does_not_return_rounded_variables(monkeypatch, est, c):
    def fail_solve(problem, solver):
        problem.status = pulp.LpStatusInfeasible
        problem.sol_status = pulp.LpSolutionNoSolutionFound
        for variable in problem.variables():
            variable.varValue = 1.0

    monkeypatch.setattr(pulp.LpProblem, "solve", fail_solve)
    with pytest.raises(RuntimeError, match="feasible integer plan: Infeasible"):
        optimize(constraints=c, est=est)


def test_planning_uses_patched_budget_and_strategy_candidates(est, truth, c, inv):
    changed = deepcopy(c)
    changed["marketing_budget"] = 10000
    strategy = SimpleNamespace(shortlist=[{"sku": "C001"}])
    result = run_planning(changed, strategy, est=est, model=truth, inv=inv, time_limit=5)
    assert result["constraints"] == changed
    assert result["optimizer"]["marketing_spend"] <= 10000
    assert result["hard_violations"] == []
    assert set(result["plan"].sku) <= {"C001", "C002", "C008", "P006", "S008", "D002"}
    assert result["violations"]
    assert changed == result["constraints"]


@pytest.mark.parametrize("exhausted", [False, True])
def test_planning_retries_validator_rejection(monkeypatch, est, truth, c, inv, exhausted):
    import src.optimizer as optimizer

    plan = pd.DataFrame(columns=["week_start", "store_id", "sku", "discount_pct"])
    solve = Mock(return_value=(plan, "Feasible"))
    monkeypatch.setattr(optimizer, "optimize", solve)
    kpis = {"marketing_spend": c["marketing_budget"] * 1.1, "promo_margin_pct": 0.25}
    monkeypatch.setattr(optimizer, "evaluate_plan", Mock(return_value=(kpis, None)))
    rejection = [{"rule": "marketing_budget", "count": 1, "example": "over budget"}]
    monkeypatch.setattr(optimizer, "check_constraints", Mock(side_effect=[rejection, rejection if exhausted else []]))
    strategy = SimpleNamespace(shortlist=[{"sku": "C008"}])
    if exhausted:
        with pytest.raises(RuntimeError, match="Plan rejected after validation"):
            run_planning(c, strategy, est=est, model=truth, inv=inv)
    else:
        result = run_planning(c, strategy, est=est, model=truth, inv=inv)
        assert result["validation_log"] == [{"attempt": 1, "rejected": rejection}]
        assert result["constraints"] == c
    assert solve.call_count == 2
    assert solve.call_args.kwargs["constraints"]["marketing_budget"] < c["marketing_budget"]


def test_orchestrator_preserves_cumulative_patches(monkeypatch, c):
    import src.agents.orchestrator as orchestrator

    extracted = SimpleNamespace(constraints=deepcopy(c))
    monkeypatch.setattr(orchestrator, "_run_input", Mock(return_value=extracted))
    relationships = Mock(return_value=[])
    monkeypatch.setattr(orchestrator, "_run_relationship", relationships)
    monkeypatch.setattr(orchestrator, "_run_strategy", Mock(return_value=SimpleNamespace(shortlist=[])))
    planning = Mock(side_effect=lambda active, strategy, **kwargs: {"constraints": deepcopy(active)})
    monkeypatch.setattr(orchestrator, "run_planning", planning)
    monkeypatch.setattr(orchestrator.Orchestrator, "_get_est", staticmethod(lambda: None))
    pipeline = orchestrator.Orchestrator()
    first = pipeline.replan({"marketing_budget": 40000})
    second = pipeline.replan({"max_discount_pct": 0.20})
    assert first.active_constraints["marketing_budget"] == 40000
    assert second.active_constraints["marketing_budget"] == 40000
    assert second.planning["constraints"]["max_discount_pct"] == 0.20
    assert extracted.constraints == c
    assert relationships.call_count == 1
    assert planning.call_count == 3
