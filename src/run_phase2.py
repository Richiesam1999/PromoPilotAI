"""Phase 2 runner: fit elasticities -> optimize -> compare with both baselines in the true world.

    python -m src.run_phase2                 # headline plan (lam = 0)
    python -m src.run_phase2 --lam 0.25      # one revenue weight
    python -m src.run_phase2 --frontier      # also sweep lam to show the revenue-vs-margin trade-off (slower)

Writes data/optimized_plan.csv and data/phase2_results.json. The true demand model is used here only as the
'real world' that scores the plan, exactly like the Phase 1 baseline; the optimizer itself never sees it.
"""
import argparse
import json

import pandas as pd

from .baseline import flat_plan
from .config import DATA
from .demand import DemandModel
from .elasticity import fit
from .evaluator import check_constraints, evaluate_plan, load_constraints, load_inventory
from .optimizer import optimize

SHOW = ["revenue", "gross_margin", "gross_margin_pct", "net_contribution", "marketing_spend", "discount_investment",
        "promoted_store_sku_weeks", "lost_sales_units", "expired_units"]


def score(plan, truth, inv, c):
    k, _ = evaluate_plan(plan, truth, inv, c)
    return k, check_constraints(plan, k, c, truth.products, truth.stores)


def fmt(key, v):
    return "-" if v is None else (f"{v:.1%}" if key.endswith("_pct") else f"{v:,.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=0.0, help="weight on revenue lift (headline: 0)")
    ap.add_argument("--frontier", action="store_true", help="sweep lam and print the revenue vs margin trade-off")
    ap.add_argument("--time-limit", type=int, default=60)
    args = ap.parse_args()

    truth, c, inv = DemandModel(), load_constraints(), load_inventory()
    est = fit()
    res = optimize(est, truth.products, truth.stores, inv, c, lam=args.lam, time_limit=args.time_limit)

    k_opt, v_opt = score(res.plan, truth, inv, c)
    k_none, v_none = score(res.plan.iloc[0:0], truth, inv, c)
    k_flat, v_flat = score(flat_plan(truth), truth, inv, c)

    print(f"solver status: {res.status} | robustness (z, clearance buffer): {res.robustness} | lam = {args.lam}")
    print(f"{'':24}{'optimized':>14}{'no promo':>14}{'flat 20%':>14}")
    for key in SHOW:
        print(f"{key:24}{fmt(key, k_opt[key]):>14}{fmt(key, k_none[key]):>14}{fmt(key, k_flat[key]):>14}")
    print(f"{'constraint rules broken':24}{len(v_opt):>14}{len(v_none):>14}{len(v_flat):>14}")
    print("sell-through (optimized / no promo / flat):")
    for key in k_opt["sell_through"]:
        t = next(x["min_sell_through_pct"] for x in c["clearance_targets"]
                 if key.startswith(x["sku"] + "@"))
        print(f"  {key:16} {k_opt['sell_through'][key]:.0%} / {k_none['sell_through'][key]:.0%} / "
              f"{k_flat['sell_through'][key]:.0%}   (target {t:.0%})")
    if v_opt:
        print("VIOLATIONS in the true world:", v_opt)
    p = res.plan
    print(f"plan: {len(p)} promoted store-SKU-weeks over {p.sku.nunique()} SKUs")
    print(p.groupby("sku").agg(rows=("sku", "size"), avg_discount=("discount_pct", "mean")).round(3).to_string())

    out = {"lam": args.lam, "robustness": res.robustness, "optimized": k_opt, "optimized_violations": v_opt,
           "no_promo": k_none, "no_promo_violations": v_none, "flat_20pct": k_flat, "flat_20pct_violations": v_flat}
    p.to_csv(DATA / "optimized_plan.csv", index=False)
    (DATA / "phase2_results.json").write_text(json.dumps(out, indent=2, default=float))

    if args.frontier:
        rows = []
        for lam in (0.0, 0.1, 0.25, 0.5, 1.0):
            r = optimize(est, truth.products, truth.stores, inv, c, lam=lam, time_limit=args.time_limit)
            k, v = score(r.plan, truth, inv, c)
            rows.append(dict(lam=lam, revenue=k["revenue"], net_contribution=k["net_contribution"],
                             spend=k["marketing_spend"], promoted_rows=len(r.plan), rules_broken=len(v)))
        fr = pd.DataFrame(rows)
        print(fr.round(0).to_string(index=False))
        fr.to_csv(DATA / "lambda_frontier.csv", index=False)


if __name__ == "__main__":
    main()
