"""Phase 4: Monte Carlo simulation.

Runs N scenarios by perturbing elasticity estimates (est.draw) and adding
Poisson demand noise (evaluate_plan noise=True), then summarises the
distribution of KPIs and constraint satisfaction rate.

Usage:
    from src.simulation import simulate, SimResult
    result = simulate(plan)          # uses optimizer plan from data/
    result.summary()                 # prints percentile table
    result.kpi_band("net_contribution")   # (p10, p50, p90)

Run standalone:  python -m src.simulation
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import DATA
from .evaluator import check_constraints, clearance_key, evaluate_plan, load_constraints, load_inventory

N_DEFAULT = 200
PERCENTILES = (5, 10, 25, 50, 75, 90, 95)
KPI_KEYS = ("revenue", "gross_margin", "gross_margin_pct", "net_contribution",
            "marketing_spend", "units_sold", "lost_sales_units", "expired_units")

# Rules that are purely structural (never affected by elasticity uncertainty).
# Clearance targets CAN fail under uncertainty even for a structurally valid plan,
# so they are tracked separately via sell_through_samples / clearance_hit_rate.
_STRUCTURAL_RULES = {
    "marketing_budget", "min_blended_promo_margin", "min_sku_margin",
    "max_discount_global", "max_discount_private_label", "no_promo_sku",
    "max_consecutive_weeks", "max_skus_per_category_per_store_week",
    "mutually_exclusive", "promote_together", "blackout",
}


def _is_structural(rule: str) -> bool:
    return any(rule == r or rule.startswith(r + ":") for r in _STRUCTURAL_RULES)


@dataclass
class SimResult:
    plan_label: str
    n_scenarios: int
    # raw arrays, shape (n_scenarios,) per KPI
    samples: dict[str, np.ndarray]
    # sell-through per clearance target, shape (n_scenarios,)
    sell_through_samples: dict[str, np.ndarray]
    # fraction of scenarios with zero STRUCTURAL violations (budget, margin, caps, etc.)
    # clearance misses under uncertainty are NOT counted here — use clearance_hit_rate()
    feasibility_rate: float
    # per-rule violation counts across all scenarios
    violation_counts: dict[str, int] = field(default_factory=dict)

    # ── Accessors ─────────────────────────────────────────────────────────────

    def kpi_band(self, key: str) -> tuple[float, float, float]:
        """Return (p10, p50, p90) for a KPI."""
        arr = self.samples[key]
        return tuple(float(np.percentile(arr, p)) for p in (10, 50, 90))

    def sell_through_band(self, target_key: str) -> tuple[float, float, float]:
        arr = self.sell_through_samples[target_key]
        return tuple(float(np.percentile(arr, p)) for p in (10, 50, 90))

    def clearance_hit_rate(self, target_key: str, threshold: float) -> float:
        """Fraction of scenarios where sell-through >= threshold."""
        return float((self.sell_through_samples[target_key] >= threshold).mean())

    def summary(self) -> str:
        lines = [
            f"\n=== Monte Carlo: {self.plan_label}  (n={self.n_scenarios}) ===",
            f"Structural feasibility rate (0 hard violations): {self.feasibility_rate:.0%}",
            "",
            f"{'KPI':<28} {'p5':>12} {'p10':>12} {'p50':>12} {'p90':>12} {'p95':>12}",
            "-" * 80,
        ]
        for key in KPI_KEYS:
            arr = self.samples[key]
            vals = [np.percentile(arr, p) for p in PERCENTILES]
            if key.endswith("_pct"):
                cells = [f"{v:.1%}" for v in vals]
            else:
                cells = [f"{v:,.0f}" for v in vals]
            # show p5, p10, p50, p90, p95 (indices 0,1,3,5,6 of PERCENTILES tuple)
            row = f"{key:<28}" + "".join(f"{cells[i]:>12}" for i in (0, 1, 3, 5, 6))
            lines.append(row)
        if self.sell_through_samples:
            lines += ["", "Sell-through:"]
            for k, arr in self.sell_through_samples.items():
                p10, p50, p90 = np.percentile(arr, [10, 50, 90])
                lines.append(f"  {k:<20} p10={p10:.0%}  p50={p50:.0%}  p90={p90:.0%}")
        if self.violation_counts:
            lines += ["", "Violation counts (across all scenarios):"]
            for rule, cnt in sorted(self.violation_counts.items(), key=lambda x: -x[1]):
                lines.append(f"  {rule:<40} {cnt:>6}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialisable summary (percentiles only, not raw arrays)."""
        out: dict = {
            "plan_label": self.plan_label,
            "n_scenarios": self.n_scenarios,
            "feasibility_rate": self.feasibility_rate,
            "violation_counts": self.violation_counts,
            "kpis": {},
            "sell_through": {},
        }
        for key in KPI_KEYS:
            arr = self.samples[key]
            out["kpis"][key] = {f"p{p}": float(np.percentile(arr, p)) for p in PERCENTILES}
        for k, arr in self.sell_through_samples.items():
            out["sell_through"][k] = {f"p{p}": float(np.percentile(arr, p)) for p in PERCENTILES}
        return out


def simulate(
    plan: pd.DataFrame,
    *,
    est=None,
    model=None,
    constraints: dict | None = None,
    inv: pd.DataFrame | None = None,
    n: int = N_DEFAULT,
    seed: int = 0,
    plan_label: str = "plan",
) -> SimResult:
    """Run n Monte Carlo scenarios for the given plan.

    Each scenario:
      1. Draws a perturbed elasticity set via est.draw(rng)
      2. Swaps E and C into the demand model (restored in finally)
      3. Runs evaluate_plan with noise=True (Poisson demand)
      4. Checks constraints; structural violations count against feasibility_rate,
         clearance misses are tracked separately in sell_through_samples.
    """
    from .demand import DemandModel
    from .elasticity import fit as fit_est
    from .evaluator import _prep_inv

    if est is None:
        est = fit_est()
    if model is None:
        model = DemandModel()
    if constraints is None:
        constraints = load_constraints()
    if inv is None:
        inv = load_inventory()

    rng = np.random.default_rng(seed)

    # Pre-cache inventory arrays (avoid repeated pandas reindex per scenario)
    inv_cache = _prep_inv(inv, model)

    # Structural violations depend only on plan structure, not elasticity draws — compute once
    _dummy_kpis = {"marketing_spend": 0, "promo_margin_pct": None, "sell_through": {}}
    structural_viols_once = [v for v in check_constraints(plan, _dummy_kpis, constraints, model.products, model.stores)
                             if _is_structural(v["rule"])]
    structural_feasible = not structural_viols_once

    kpi_lists: dict[str, list] = {k: [] for k in KPI_KEYS}
    st_lists: dict[str, list] = {}
    violation_counts: dict[str, int] = {v["rule"]: 0 for v in structural_viols_once}
    structural_feasible_count = 0

    for _ in range(n):
        drawn = est.draw(rng)
        orig_E, orig_C = model.E.copy(), model.C.copy()
        model.E = drawn.E
        model.C = drawn.C
        try:
            kpis, _ = evaluate_plan(plan, model, inv, constraints, noise=True, rng=rng,
                                    _inv_cache=inv_cache)
        finally:
            model.E = orig_E
            model.C = orig_C

        for k in KPI_KEYS:
            kpi_lists[k].append(kpis[k])
        for key, val in kpis.get("sell_through", {}).items():
            st_lists.setdefault(key, []).append(val)

        # Only re-check the KPI-dependent rules (budget, margin) per scenario
        spend_viol = kpis["marketing_spend"] > constraints["marketing_budget"]
        pm = kpis["promo_margin_pct"]
        margin_viol = pm is not None and pm < constraints["min_blended_promo_margin_pct"] - 1e-9
        scenario_structural_ok = structural_feasible and not spend_viol and not margin_viol
        if scenario_structural_ok:
            structural_feasible_count += 1
        if spend_viol:
            violation_counts["marketing_budget"] = violation_counts.get("marketing_budget", 0) + 1
        if margin_viol:
            violation_counts["min_blended_promo_margin"] = violation_counts.get("min_blended_promo_margin", 0) + 1
        for v in structural_viols_once:
            violation_counts[v["rule"]] = violation_counts.get(v["rule"], 0) + 1
        # Track clearance violations
        for tg in constraints.get("clearance_targets", []):
            key = clearance_key(tg)
            got = kpis["sell_through"].get(key)
            if got is not None and got < tg["min_sell_through_pct"] - 1e-9:
                violation_counts[f"clearance:{key}"] = violation_counts.get(f"clearance:{key}", 0) + 1

    return SimResult(
        plan_label=plan_label,
        n_scenarios=n,
        samples={k: np.array(v) for k, v in kpi_lists.items()},
        sell_through_samples={k: np.array(v) for k, v in st_lists.items()},
        feasibility_rate=structural_feasible_count / n,
        violation_counts=violation_counts,
    )


def simulate_scenarios(
    plans: dict[str, pd.DataFrame],
    **kwargs,
) -> dict[str, SimResult]:
    """Simulate multiple named plans and return results keyed by label."""
    return {label: simulate(plan, plan_label=label, **kwargs) for label, plan in plans.items()}


# ── Edge-case helpers (used by tests and demo) ────────────────────────────────

def zero_stock_plan(model) -> pd.DataFrame:
    """A plan that promotes only SKUs with zero on-hand inventory (stress test)."""
    inv = load_inventory()
    zero_skus = inv[inv.on_hand_units == 0].sku.unique()
    if len(zero_skus) == 0:
        return pd.DataFrame(columns=["week_start", "store_id", "sku", "discount_pct"])
    from .config import plan_weeks
    weeks = plan_weeks()
    rows = [{"week_start": weeks[0], "store_id": s, "sku": k, "discount_pct": 0.20}
            for s in model.store_ids for k in zero_skus[:2]]
    return pd.DataFrame(rows)


def impossible_budget_constraints(base: dict) -> dict:
    """Return a copy of constraints with budget set to $1 (impossible)."""
    import copy
    c = copy.deepcopy(base)
    c["marketing_budget"] = 1
    return c


def conflicting_constraints(base: dict) -> dict:
    """Return constraints where the same SKU is both no_promo and has a clearance target."""
    import copy
    c = copy.deepcopy(base)
    c["clearance_targets"] = c.get("clearance_targets", []) + [
        {"sku": "D001", "stores": "all", "min_sell_through_pct": 0.50}
    ]
    return c


def main():
    from .baseline import flat_plan
    from .demand import DemandModel
    from .elasticity import fit

    model = DemandModel()
    est = fit()
    c = load_constraints()
    inv = load_inventory()

    opt_path = DATA / "optimizer_plan.csv"
    if opt_path.exists():
        opt_plan = pd.read_csv(opt_path, parse_dates=["week_start"])
    else:
        from .config import plan_weeks
        W = plan_weeks()
        rows = []
        for sku, stores, d, weeks in [
            ("C008", model.store_ids, 0.30, [0, 1, 2, 4]),
            ("P006", model.store_ids, 0.20, [0, 1]),
            ("S008", model.store_ids, 0.25, [0, 1, 2]),
            ("D002", ["S03", "S05"], 0.15, [0, 1]),
        ]:
            for wi in weeks:
                for s in stores:
                    rows.append({"week_start": W[wi], "store_id": s, "sku": sku, "discount_pct": d})
        opt_plan = pd.DataFrame(rows)

    plans = {
        "optimizer": opt_plan,
        "flat_20pct": flat_plan(model),
        "no_promo": opt_plan.iloc[0:0],
    }

    results = simulate_scenarios(plans, est=est, model=model, constraints=c, inv=inv, n=200)

    for label, res in results.items():
        print(res.summary())

    out = {label: res.to_dict() for label, res in results.items()}
    (DATA / "simulation_results.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nwritten to {DATA / 'simulation_results.json'}")


if __name__ == "__main__":
    main()
