"""Phase 3 – Strategy Agent.

Given extracted constraints, relationship groups, and inventory/elasticity data,
the LLM shortlists SKUs to promote, picks a mechanism (% off / BOGO / bundle),
and writes a rationale for each decision.

Usage:
    from src.agents.strategy_agent import run
    strategy = run(constraints, groups)   # StrategyResult

Run standalone:  python -m src.agents.strategy_agent
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd

from ..config import DATA, GOLD
from ._llm import call_json

_SYSTEM = """You are a retail promotion strategist.
You will receive:
1. constraints – the extracted promotion constraints (budget, margin floors, blackouts, clearance targets, etc.)
2. relationship_groups – substitute and complement clusters identified by the relationship agent
3. inventory_pressure – SKUs with high stock or expiry risk that need clearance
4. elasticity_summary – own-price elasticities per SKU (more negative = more price-sensitive)

Your task: produce a promotion strategy.

Return ONLY a JSON object:
{
  "shortlist": [
    {
      "sku": "<sku_id>",
      "name": "<product name>",
      "mechanism": "pct_off" | "bogo" | "bundle",
      "suggested_discount_pct": <float 0-1>,
      "priority": "must_promote" | "high" | "medium" | "low",
      "rationale": "<one sentence>",
      "cannibalization_warning": "<sku to watch, or null>"
    },
    ...
  ],
  "bundle_suggestions": [
    {"skus": ["<sku_a>", "<sku_b>"], "rationale": "<one sentence>"}
  ],
  "excluded_skus": [
    {"sku": "<sku_id>", "reason": "<why excluded>"}
  ],
  "overall_rationale": "<2-3 sentence summary of the strategy>"
}

Rules:
- must_promote: any SKU with a clearance target in the constraints.
- Respect all constraints: no_promo_skus must appear in excluded_skus, brand discount caps must be
  reflected in suggested_discount_pct, blackout SKUs must be excluded for the relevant region.
- For substitutes: if both are shortlisted, flag cannibalization_warning on the lower-priority one.
- For complements: suggest a bundle if both are high-margin and basket affinity is strong.
- Prefer pct_off for clearance (simple, measurable); BOGO for high-volume staples; bundle for complements.
- Do not suggest discounts that would breach min_sku_margin_pct or brand caps.
- Keep the shortlist focused: aim for 8-15 SKUs, not all 48.
"""


@dataclass
class StrategyResult:
    shortlist: list[dict]
    bundle_suggestions: list[dict]
    excluded_skus: list[dict]
    overall_rationale: str
    raw: dict


def _inventory_pressure(top_n: int = 12) -> pd.DataFrame:
    """Return SKUs with highest stock relative to baseline weekly sales (clearance pressure)."""
    inv = pd.read_csv(DATA / "inventory.csv")
    products = pd.read_csv(DATA / "products.csv")
    sales = pd.read_csv(DATA / "sales_history.csv")
    avg_weekly = sales.groupby("sku").units.mean().rename("avg_weekly_units")
    stock = inv.groupby("sku").on_hand_units.sum().rename("total_stock")
    # days_to_expiry lives in inventory.csv (min across stores per SKU)
    expiry = inv.groupby("sku").days_to_expiry.min().rename("days_to_expiry")
    df = pd.concat([stock, avg_weekly, expiry], axis=1).dropna(subset=["total_stock", "avg_weekly_units"])
    df["weeks_cover"] = df.total_stock / df.avg_weekly_units.clip(lower=1)
    df = df.reset_index().merge(products[["sku", "name", "category"]], on="sku", how="left")
    return df.nlargest(top_n, "weeks_cover")[["sku", "name", "category", "total_stock",
                                              "avg_weekly_units", "weeks_cover", "days_to_expiry"]]


def _elasticity_summary(est) -> pd.DataFrame:
    products = pd.read_csv(DATA / "products.csv")
    df = pd.DataFrame({"sku": est.skus, "own_elasticity": est.E})
    return df.merge(products[["sku", "name", "category"]], on="sku")


def run(constraints: dict | None = None,
        groups=None,
        est=None) -> StrategyResult:
    if constraints is None:
        constraints = json.loads((GOLD / "constraints.json").read_text())
    if groups is None:
        from .relationship_agent import run as rel_run
        groups = rel_run(est=est)
    if est is None:
        from ..elasticity import fit
        est = fit()

    inv_df = _inventory_pressure()
    elas_df = _elasticity_summary(est)

    groups_payload = [
        {"group_id": g.group_id, "name": g.name, "relation": g.relation,
         "skus": g.skus, "planning_note": g.planning_note,
         "cannibalization_risk": g.cannibalization_risk}
        for g in groups
    ]

    user_msg = (
        f"constraints:\n{json.dumps(constraints, indent=2)}\n\n"
        f"relationship_groups:\n{json.dumps(groups_payload, indent=2)}\n\n"
        f"inventory_pressure (top 12 by weeks of cover):\n{inv_df.to_csv(index=False)}\n\n"
        f"elasticity_summary:\n{elas_df.to_csv(index=False)}"
    )

    raw = call_json(_SYSTEM, user_msg)

    # Post-process: guarantee all clearance-target SKUs are marked must_promote.
    # The LLM occasionally misses one; this is a deterministic safety net.
    clearance_skus = {t["sku"] for t in constraints.get("clearance_targets", [])}
    shortlist = raw.get("shortlist", [])
    shortlist_skus = {s["sku"] for s in shortlist}
    for s in shortlist:
        if s["sku"] in clearance_skus:
            s["priority"] = "must_promote"
    # Add any clearance SKU the LLM omitted entirely
    products_df = pd.read_csv(DATA / "products.csv").set_index("sku")
    for sku in clearance_skus:
        if sku not in shortlist_skus and sku in products_df.index:
            shortlist.append({
                "sku": sku,
                "name": products_df.loc[sku, "name"],
                "mechanism": "pct_off",
                "suggested_discount_pct": 0.20,
                "priority": "must_promote",
                "rationale": "Clearance target — added by post-processor.",
                "cannibalization_warning": None,
            })
    raw["shortlist"] = shortlist

    return StrategyResult(
        shortlist=raw.get("shortlist", []),
        bundle_suggestions=raw.get("bundle_suggestions", []),
        excluded_skus=raw.get("excluded_skus", []),
        overall_rationale=raw.get("overall_rationale", ""),
        raw=raw,
    )


def main():
    result = run()
    print(f"\n=== Strategy Agent ===")
    print(f"Overall: {result.overall_rationale}\n")
    print(f"Shortlist ({len(result.shortlist)} SKUs):")
    for s in result.shortlist:
        warn = f"  ⚠ cannibalization: {s['cannibalization_warning']}" if s.get("cannibalization_warning") else ""
        print(f"  [{s['priority']}] {s['sku']} {s['name']}  {s['mechanism']} "
              f"{s['suggested_discount_pct']:.0%}  — {s['rationale']}{warn}")
    if result.bundle_suggestions:
        print(f"\nBundle suggestions:")
        for b in result.bundle_suggestions:
            print(f"  {'+'.join(b['skus'])}: {b['rationale']}")
    print(f"\nExcluded ({len(result.excluded_skus)}):")
    for e in result.excluded_skus:
        print(f"  {e['sku']}: {e['reason']}")


if __name__ == "__main__":
    main()
