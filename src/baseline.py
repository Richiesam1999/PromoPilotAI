"""Flat-20%-off-everything baseline. Run:  python -m src.baseline"""
import json

import pandas as pd

from .config import BASELINE_DISCOUNT, DATA, GOLD, plan_weeks
from .demand import DemandModel
from .evaluator import check_constraints, evaluate_plan, load_constraints, load_inventory


def flat_plan(model, discount=BASELINE_DISCOUNT):
    idx = pd.MultiIndex.from_product([plan_weeks(), model.store_ids, model.skus],
                                     names=["week_start", "store_id", "sku"])
    return idx.to_frame(index=False).assign(discount_pct=discount)


def run_baseline(model=None, constraints=None):
    model = model or DemandModel()
    inv = load_inventory()
    c = constraints or (load_constraints() if (GOLD / "constraints.json").exists() else None)
    plan = flat_plan(model)
    kpis, _ = evaluate_plan(plan, model, inv, c)
    viol = check_constraints(plan, kpis, c, model.products, model.stores) if c else []
    return plan, kpis, viol


def main():
    model = DemandModel()
    inv = load_inventory()
    c = load_constraints()
    none_k, _ = evaluate_plan(flat_plan(model, 0.0).iloc[0:0], model, inv, c)
    plan, k, viol = run_baseline(model, c)
    out = {"no_promo": none_k, "flat_20pct": k, "flat_20pct_violations": viol}
    (DATA / "baseline_results.json").write_text(json.dumps(out, indent=2, default=float))
    f = lambda d: (f"rev ${d['revenue']:,.0f} | GM ${d['gross_margin']:,.0f} ({d['gross_margin_pct']:.1%}) | "
                   f"net contrib ${d['net_contribution']:,.0f} | spend ${d['marketing_spend']:,.0f} | "
                   f"lost units {d['lost_sales_units']:,.0f} | expired {d['expired_units']:,.0f}")
    print("NO PROMO :", f(none_k)); print("FLAT 20% :", f(k))
    print("sell-through:", {a: f"{b:.0%}" for a, b in k["sell_through"].items()})
    print(f"constraint violations ({len(viol)} rules broken):")
    for v in viol:
        print(f"  - {v['rule']}: {v['count']}  {v['example']}")


if __name__ == "__main__":
    main()
