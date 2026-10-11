"""Phase 3 – Orchestrator.

Chains Input → Relationship → Strategy → optimizer → evaluator → validator. Caches each stage result so that
when a single constraint changes only the affected downstream stages are re-run.

Dependency graph:
    input_agent  ──►  relationship_agent  ──►  strategy_agent
                  └──────────────────────────►  strategy_agent

Re-planning rules:
    - constraints change  → re-run strategy, optimizer, evaluator and validator
  - new text inputs     → re-run all three stages
  - elasticity changes  → re-run relationship_agent + strategy_agent

Usage:
    from src.agents.orchestrator import Orchestrator
    orch = Orchestrator()
    plan = orch.run()           # full pipeline
    plan = orch.replan({"marketing_budget": 45000})   # patch and validate a new plan
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from ..optimizer import run_planning
from .input_agent import InputResult, run as _run_input
from .relationship_agent import RelationshipGroup, run as _run_relationship
from .strategy_agent import StrategyResult, run as _run_strategy


@dataclass
class PipelineResult:
    input_result: InputResult
    groups: list[RelationshipGroup]
    strategy: StrategyResult
    replan_log: list[str] = field(default_factory=list)
    planning: dict | None = None
    active_constraints: dict = field(default_factory=dict)


class Orchestrator:
    """Stateful orchestrator.  Caches stage outputs and re-runs minimally on changes."""

    def __init__(self):
        self._input_result: InputResult | None = None
        self._groups: list[RelationshipGroup] | None = None
        self._strategy: StrategyResult | None = None
        self._est = None   # cached elasticity estimates (expensive to refit)
        self._constraints = None
        self._planning = None

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> PipelineResult:
        """Extract, shortlist, solve, evaluate and validate a promotion plan."""
        self._est = self._get_est()
        self._input_result = _run_input()
        self._constraints = copy.deepcopy(self._input_result.constraints)
        self._groups = _run_relationship(est=self._est)
        self._strategy = _run_strategy(
            constraints=self._constraints,
            groups=self._groups,
            est=self._est,
        )
        self._planning = run_planning(self._constraints, self._strategy, est=self._est)
        return PipelineResult(
            input_result=self._input_result,
            groups=self._groups,
            strategy=self._strategy,
            planning=self._planning,
            active_constraints=copy.deepcopy(self._constraints),
        )

    def replan(self, constraint_overrides: dict[str, Any]) -> PipelineResult:
        """Patch active constraints and re-run strategy plus deterministic planning.

        If no prior run exists, falls back to a full run first.
        """
        if self._input_result is None:
            self.run()

        log = []
        patched = copy.deepcopy(self._constraints)
        for key, val in constraint_overrides.items():
            old = patched.get(key)
            patched[key] = val
            log.append(f"constraint '{key}' changed: {old!r} → {val!r}")

        strategy = _run_strategy(
            constraints=patched,
            groups=self._groups,
            est=self._est,
        )
        planning = run_planning(patched, strategy, est=self._est)
        self._strategy = strategy
        self._planning = planning
        self._constraints = patched
        log.append("strategy_agent re-run; optimizer, evaluator and validator re-run; input_agent and relationship_agent results reused from cache")

        return PipelineResult(
            input_result=self._input_result,
            groups=self._groups,
            strategy=self._strategy,
            replan_log=log,
            planning=self._planning,
            active_constraints=copy.deepcopy(self._constraints),
        )

    def rerun_relationships(self) -> PipelineResult:
        """Re-run relationship + strategy (e.g. after elasticity model update)."""
        if self._input_result is None:
            return self.run()
        self._est = self._get_est()
        self._groups = _run_relationship(est=self._est)
        self._strategy = _run_strategy(
            constraints=self._constraints,
            groups=self._groups,
            est=self._est,
        )
        self._planning = run_planning(self._constraints, self._strategy, est=self._est)
        return PipelineResult(
            input_result=self._input_result,
            groups=self._groups,
            strategy=self._strategy,
            planning=self._planning,
            active_constraints=copy.deepcopy(self._constraints),
            replan_log=["relationship_agent and strategy_agent re-run; input_agent result reused"],
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _get_est():
        from ..elasticity import fit
        return fit()


def main():
    orch = Orchestrator()
    print("Running full pipeline...")
    result = orch.run()

    print(f"\n── Input Agent ──")
    print(f"  Budget extracted: ${result.input_result.constraints.get('marketing_budget'):,}")
    print(f"  Warnings: {len(result.input_result.warnings)}")
    for w in result.input_result.warnings:
        print(f"    - {w}")

    print(f"\n── Relationship Agent ──")
    for g in result.groups:
        print(f"  [{g.group_id}] {g.name}  ({g.relation})  SKUs: {', '.join(g.skus)}")

    print(f"\n── Strategy Agent ──")
    print(f"  {result.strategy.overall_rationale}")
    print(f"  Shortlist: {[s['sku'] for s in result.strategy.shortlist]}")

    print(f"\n── Replan demo: cut budget by 20% ──")
    new_budget = int(result.input_result.constraints.get("marketing_budget", 59000) * 0.80)
    replan = orch.replan({"marketing_budget": new_budget})
    for entry in replan.replan_log:
        print(f"  {entry}")
    print(f"  New shortlist: {[s['sku'] for s in replan.strategy.shortlist]}")
    print(f"  Solver: {replan.planning['status']}")
    print(f"  Spend: ${replan.planning['optimizer']['marketing_spend']:,.0f}")


if __name__ == "__main__":
    main()
