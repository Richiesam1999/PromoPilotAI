"""Phase 2b: promotion plan optimizer.

Objective: maximise  net_contribution + lambda_rev * revenue_lift
           minus a heavy clearance-shortfall penalty, subject to hard policy constraints.

Decision variables: x[w, s, k, d] in {0, 1}  (promote SKU k in store s in week w at discount depth d)
  - at most one discount depth active per (w, s, k)
  - discount grid: DISCOUNT_GRID (subset of allowed depths)

Run:  python -m src.optimizer
"""
import json
from copy import deepcopy

import numpy as np
import pandas as pd
import pulp

from .config import DATA, GOLD, plan_weeks
from .elasticity import fit as fit_elasticity
from .evaluator import check_constraints, clearance_key, evaluate_plan, load_constraints, load_inventory

DISCOUNT_GRID = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
LAMBDA_REV = 0.10          # weight on revenue lift vs net contribution (expose as slider in UI)
BIG_M = 1e9
# LP linearisation ignores cross-elasticity uplift so true model always sells more than LP predicts.
# Per-SKU buffers compensate: LP target = constraint target * buffer.
# D002 has a strong milk complement (C=-0.5) that the LP misses entirely.
CLEARANCE_BUFFER = {"D002": 1.12, "default": 1.02}
CLEARANCE_PENALTY = 1000.0


def _build_plan(x_vals, weeks, store_ids, skus, discount_grid):
    rows = []
    for wi, ws in enumerate(weeks):
        for si, st in enumerate(store_ids):
            for ki, sku in enumerate(skus):
                for di, d in enumerate(discount_grid):
                    if x_vals.get((wi, si, ki, di), 0) > 0.5:
                        rows.append({"week_start": ws, "store_id": st, "sku": sku, "discount_pct": d})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["week_start", "store_id", "sku", "discount_pct"])


def optimize(constraints=None, lambda_rev=LAMBDA_REV, discount_grid=None, est=None, verbose=False,
             time_limit=120, candidate_skus=None):
    discount_grid = discount_grid or DISCOUNT_GRID
    c = constraints or load_constraints()
    inv_df = load_inventory()
    est = est or fit_elasticity()

    weeks = plan_weeks()
    store_ids = est.store_ids
    skus = est.skus
    W, S, K, D = len(weeks), len(store_ids), len(skus), len(discount_grid)

    products = pd.read_csv(DATA / "products.csv")
    stores = pd.read_csv(DATA / "stores.csv")
    lp = products.list_price.to_numpy(float)          # (K,)
    uc = products.unit_cost.to_numpy(float)           # (K,)
    brand = products.brand.to_numpy()
    subcat = products.subcategory.to_numpy()
    category = products.category.to_numpy()
    is_pl = products.is_private_label.to_numpy(int)
    region = stores.region.to_numpy()
    sku_idx = {s: i for i, s in enumerate(skus)}
    store_idx = {s: i for i, s in enumerate(store_ids)}

    # Pre-compute expected units and margins for each (w, s, k, d)
    ones = np.ones((S, K))
    base_units = np.stack([est.baseline_units(ws) for ws in weeks])   # (W, S, K)

    # For cross-elasticity: compute uplift ratio for each (w, s, k, d) ignoring cross effects
    # (cross effects are captured approximately via the objective; exact cross would require
    #  quadratic terms — we linearise by using the estimated own-price uplift only in the LP,
    #  and verify the full cross-elasticity effect post-solve via evaluate_plan)
    uplift = np.zeros((W, S, K, D))
    for di, d in enumerate(discount_grid):
        r = ones * (1 - d)
        for wi, ws in enumerate(weeks):
            uplift[wi, :, :, di] = est.expected_units(ws, r) / np.maximum(base_units[wi], 1e-6)

    sold = base_units[:, :, :, None] * uplift          # (W, S, K, D) units if promoted at depth d
    rev_promo = sold * lp[None, None, :, None] * (1 - np.array(discount_grid))[None, None, None, :]
    gm_promo = rev_promo - sold * uc[None, None, :, None]
    exec_cost = 15.0   # per store-sku-week (from config)

    # ── PuLP model ────────────────────────────────────────────────────────────
    prob = pulp.LpProblem("promo_optimizer", pulp.LpMaximize)
    x = pulp.LpVariable.dicts("x", ((w, s, k, d) for w in range(W) for s in range(S)
                                     for k in range(K) for d in range(D)), cat="Binary")

    if candidate_skus is not None:
        eligible = set(candidate_skus) | {target["sku"] for target in c.get("clearance_targets", [])}
        while True:
            expanded = eligible | {sku for pair in c.get("promote_together", [])
                                   if eligible.intersection(pair) for sku in pair}
            if expanded == eligible:
                break
            eligible = expanded
        for ki, sku in enumerate(skus):
            if sku not in eligible:
                for w in range(W):
                    for s in range(S):
                        for d in range(D):
                            prob += x[w, s, ki, d] == 0

    # at most one depth per (w, s, k)
    for w in range(W):
        for s in range(S):
            for k in range(K):
                prob += pulp.lpSum(x[w, s, k, d] for d in range(D)) <= 1

    # ── Hard constraints ──────────────────────────────────────────────────────

    # no_promo_skus
    for sku in c.get("no_promo_skus", []):
        if sku in sku_idx:
            k = sku_idx[sku]
            for w in range(W):
                for s in range(S):
                    for d in range(D):
                        prob += x[w, s, k, d] == 0

    # max_discount_pct
    max_d_global = c.get("max_discount_pct", 0.35)
    for di, d in enumerate(discount_grid):
        if d > max_d_global + 1e-9:
            for w in range(W):
                for s in range(S):
                    for k in range(K):
                        prob += x[w, s, k, di] == 0

    # max_discount_by_brand
    for brand_name, cap in c.get("max_discount_by_brand", {}).items():
        for ki, sku in enumerate(skus):
            if brand[ki] == brand_name:
                for di, d in enumerate(discount_grid):
                    if d > cap + 1e-9:
                        for w in range(W):
                            for s in range(S):
                                prob += x[w, s, ki, di] == 0

    # max_discount_private_label
    pl_cap = c.get("max_discount_private_label", 0.30)
    for ki in range(K):
        if is_pl[ki]:
            for di, d in enumerate(discount_grid):
                if d > pl_cap + 1e-9:
                    for w in range(W):
                        for s in range(S):
                            prob += x[w, s, ki, di] == 0

    # min_sku_margin_pct: price*(1-d) >= uc*(1+min_margin/(1-min_margin)) => d <= 1 - uc/lp/(1-min_m)
    min_sku_m = c.get("min_sku_margin_pct", 0.12)
    for ki in range(K):
        max_allowed = 1.0 - uc[ki] / lp[ki] / (1 - min_sku_m) if lp[ki] > 0 else 0.0
        for di, d in enumerate(discount_grid):
            if d > max_allowed + 1e-9:
                for w in range(W):
                    for s in range(S):
                        prob += x[w, s, ki, di] == 0

    # blackouts
    for bo in c.get("blackouts", []):
        ki = sku_idx.get(bo["sku"])
        if ki is None:
            continue
        until = pd.Timestamp(bo["until"])
        for wi, ws in enumerate(weeks):
            if ws < until:
                for si, st in enumerate(store_ids):
                    if region[si] == bo["region"]:
                        for d in range(D):
                            prob += x[wi, si, ki, d] == 0

    # mutually_exclusive_same_week
    for a, b in c.get("mutually_exclusive_same_week", []):
        ka, kb = sku_idx.get(a), sku_idx.get(b)
        if ka is None or kb is None:
            continue
        for w in range(W):
            for s in range(S):
                prob += (pulp.lpSum(x[w, s, ka, d] for d in range(D)) +
                         pulp.lpSum(x[w, s, kb, d] for d in range(D))) <= 1

    # promote_together
    for a, b in c.get("promote_together", []):
        ka, kb = sku_idx.get(a), sku_idx.get(b)
        if ka is None or kb is None:
            continue
        for w in range(W):
            for s in range(S):
                xa = pulp.lpSum(x[w, s, ka, d] for d in range(D))
                xb = pulp.lpSum(x[w, s, kb, d] for d in range(D))
                prob += xa == xb

    # max_consecutive_promo_weeks (limit L: no L+1 consecutive active weeks)
    L = c.get("max_consecutive_promo_weeks", 3)
    if W > L:
        for s in range(S):
            for k in range(K):
                for w in range(W - L):
                    prob += pulp.lpSum(
                        x[w + t, s, k, d] for t in range(L + 1) for d in range(D)
                    ) <= L

    # max_promoted_skus_per_category_per_store_week
    cat_limit = c.get("max_promoted_skus_per_category_per_store_week", 3)
    cats_unique = list(dict.fromkeys(category))
    for w in range(W):
        for s in range(S):
            for cat in cats_unique:
                ks_in_cat = [ki for ki, c_ in enumerate(category) if c_ == cat]
                prob += pulp.lpSum(
                    x[w, s, ki, d] for ki in ks_in_cat for d in range(D)
                ) <= cat_limit

    # marketing_budget: sum of (discount_investment + execution_cost) <= budget
    budget = c["marketing_budget"]
    spend_terms = []
    for w in range(W):
        for s in range(S):
            for ki in range(K):
                for di, d in enumerate(discount_grid):
                    disc_inv = float(sold[w, s, ki, di]) * lp[ki] * d
                    spend_terms.append((disc_inv + exec_cost) * x[w, s, ki, di])
    prob += pulp.lpSum(spend_terms) <= budget

    # min_blended_promo_margin_pct: sum(gm_promo * x) >= min_pct * sum(rev_promo * x)
    min_pm = c.get("min_blended_promo_margin_pct", 0.21)
    gm_terms = [float(gm_promo[w, s, ki, di]) * x[w, s, ki, di]
                for w in range(W) for s in range(S) for ki in range(K) for di in range(D)]
    rev_terms = [float(rev_promo[w, s, ki, di]) * x[w, s, ki, di]
                 for w in range(W) for s in range(S) for ki in range(K) for di in range(D)]
    prob += pulp.lpSum(gm_terms) >= min_pm * pulp.lpSum(rev_terms)

    # clearance targets: penalized unit slack + minimum discount depth
    # Minimum depth prevents the LP satisfying the unit constraint with trivial 10% discounts
    # that the true model won't honour (LP ignores cross-elasticity uplift).
    # Expiry-aware: block promotion after the expiry week so the LP doesn't waste budget on
    # weeks where the evaluator will zero out stock.
    # Per-SKU max allowed discount (from min_sku_margin_pct already applied above).
    # Clearance floor = min(max_allowed, 0.20) so we never breach the margin constraint.
    clearance_slacks = {}
    inv_raw = load_inventory()
    inv_idx = inv_raw.set_index(["store_id", "sku"])
    for tg in c.get("clearance_targets", []):
        ki = sku_idx.get(tg["sku"])
        if ki is None:
            continue
        target_stores = store_ids if tg["stores"] == "all" else tg["stores"]
        target_si = [store_idx[st] for st in target_stores if st in store_idx]
        total_stock = sum(int(inv_idx.loc[(store_ids[si], skus[ki]), "on_hand_units"])
                          for si in target_si if (store_ids[si], skus[ki]) in inv_idx.index)
        min_sold = tg["min_sell_through_pct"] * total_stock
        # Determine expiry cutoff: earliest expiry across target stores for this SKU
        expiry_days_list = [
            inv_idx.loc[(store_ids[si], skus[ki]), "days_to_expiry"]
            for si in target_si if (store_ids[si], skus[ki]) in inv_idx.index
        ]
        expiry_days_list = [e for e in expiry_days_list if not pd.isna(e)]
        if expiry_days_list:
            expiry_cutoff = weeks[0] + pd.Timedelta(days=int(min(expiry_days_list)))
            valid_weeks = [wi for wi, ws in enumerate(weeks) if ws < expiry_cutoff]
        else:
            valid_weeks = list(range(W))
        # Block promotion after expiry
        for wi in range(W):
            if wi not in valid_weeks:
                for si in target_si:
                    for di in range(D):
                        prob += x[wi, si, ki, di] == 0
        base_sold = sum(float(base_units[w, si, ki]) for w in valid_weeks for si in target_si)
        buf = CLEARANCE_BUFFER.get(tg["sku"], CLEARANCE_BUFFER["default"])
        extra_needed = max(0.0, min_sold * buf - base_sold)
        slack = pulp.LpVariable(f"clearance_shortfall_{len(clearance_slacks)}", lowBound=0)
        clearance_slacks[clearance_key(tg)] = slack
        extra_terms = [
            float(base_units[w, si, ki]) * (float(uplift[w, si, ki, di]) - 1.0) * x[w, si, ki, di]
            for w in valid_weeks for si in target_si for di in range(D)
        ]
        prob += pulp.lpSum(extra_terms) + slack >= extra_needed
        # Minimum discount depth: use the deepest depth that still respects min_sku_margin
        max_allowed_d = 1.0 - uc[ki] / lp[ki] / (1 - min_sku_m) if lp[ki] > 0 else 0.0
        floor_d = min(0.20, max_allowed_d - 1e-9)  # never exceed margin cap
        if floor_d > discount_grid[0] + 1e-9:
            # last grid point that does NOT exceed the margin cap
            floor_di = max((di for di, d in enumerate(discount_grid) if d <= floor_d + 1e-9), default=0)
            for w in valid_weeks:
                for si in target_si:
                    prob += pulp.lpSum(x[w, si, ki, di] for di in range(floor_di)) == 0

    # ── Objective ─────────────────────────────────────────────────────────────
    # net_contribution proxy = gm_promo - exec_cost; revenue lift = rev_promo - base_rev
    base_rev_total = float(base_units.sum() * np.mean(lp))   # rough scale normaliser
    obj_terms = []
    for w in range(W):
        for s in range(S):
            for ki in range(K):
                for di in range(D):
                    nc = float(gm_promo[w, s, ki, di]) - exec_cost
                    rev_lift = float(rev_promo[w, s, ki, di]) - float(base_units[w, s, ki]) * lp[ki]
                    obj_terms.append((nc + lambda_rev * rev_lift) * x[w, s, ki, di])
    prob += pulp.lpSum(obj_terms) - CLEARANCE_PENALTY * pulp.lpSum(clearance_slacks.values())

    solver = pulp.PULP_CBC_CMD(msg=1 if verbose else 0, timeLimit=time_limit)
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    if prob.sol_status not in (pulp.LpSolutionOptimal, pulp.LpSolutionIntegerFeasible):
        raise RuntimeError(f"Optimizer did not return a feasible integer plan: {status}")
    if prob.sol_status == pulp.LpSolutionIntegerFeasible:
        status = "Feasible"
    x_vals = {(w, s, k, d): pulp.value(x[w, s, k, d]) or 0
              for w in range(W) for s in range(S) for k in range(K) for d in range(D)}
    plan = _build_plan(x_vals, weeks, store_ids, skus, discount_grid)
    plan.attrs["clearance_slack_units"] = {
        key: float(pulp.value(slack) or 0.0) for key, slack in clearance_slacks.items()
    }
    plan.attrs["solution_status"] = pulp.LpSolution[prob.sol_status]
    return plan, status


def run_planning(constraints, strategy, est=None, model=None, inv=None, time_limit=15):
    from .baseline import flat_plan
    from .demand import DemandModel

    active = deepcopy(constraints)
    solving = deepcopy(active)
    model = model if model is not None else DemandModel()
    inv = inv if inv is not None else load_inventory()
    est = est if est is not None else fit_elasticity()
    candidates = [item["sku"] for item in strategy.shortlist]
    validation_log = []
    for attempt in range(2):
        plan, status = optimize(constraints=solving, est=est, candidate_skus=candidates,
                                time_limit=time_limit)
        kpis, _ = evaluate_plan(plan, model, inv, active)
        violations = check_constraints(plan, kpis, active, model.products, model.stores)
        hard = [violation for violation in violations if not violation["rule"].startswith("clearance:")]
        if not hard:
            break
        validation_log.append({"attempt": attempt + 1, "rejected": hard})
        rules = {violation["rule"] for violation in hard}
        if attempt == 1 or rules - {"marketing_budget", "min_blended_promo_margin"}:
            raise RuntimeError(f"Plan rejected after validation: {hard}")
        if "marketing_budget" in rules:
            solving["marketing_budget"] *= active["marketing_budget"] / kpis["marketing_spend"] * 0.95
        if "min_blended_promo_margin" in rules:
            solving["min_blended_promo_margin_pct"] += (
                active["min_blended_promo_margin_pct"] - kpis["promo_margin_pct"] + 0.02
            )
    flat_kpis, _ = evaluate_plan(flat_plan(model), model, inv, active)
    none_kpis, _ = evaluate_plan(plan.iloc[0:0], model, inv, active)
    return {
        "plan": plan, "constraints": active, "strategy": strategy,
        "status": status, "solution_status": plan.attrs.get("solution_status", ""),
        "optimizer": kpis, "flat_20pct": flat_kpis, "no_promo": none_kpis,
        "violations": violations, "hard_violations": hard,
        "clearance_slack_units": plan.attrs.get("clearance_slack_units", {}),
        "validation_log": validation_log,
    }


def main():
    from .demand import DemandModel
    from .baseline import flat_plan, run_baseline

    c = load_constraints()
    inv = load_inventory()
    model = DemandModel()

    print("Running optimizer...")
    plan, status = optimize(constraints=c)
    print(f"Solver status: {status}")

    kpis, _ = evaluate_plan(plan, model, inv, c)
    viol = check_constraints(plan, kpis, c, model.products, model.stores)
    hard_violations = [v for v in viol if not v["rule"].startswith("clearance:")]
    if hard_violations:
        raise RuntimeError(f"Optimizer plan failed hard-rule validation: {hard_violations}")

    _, base_kpis, base_viol = run_baseline(model, c)
    none_kpis, _ = evaluate_plan(flat_plan(model, 0.0).iloc[0:0], model, inv, c)

    f = lambda d: (f"rev ${d['revenue']:,.0f} | GM ${d['gross_margin']:,.0f} ({d['gross_margin_pct']:.1%}) | "
                   f"net contrib ${d['net_contribution']:,.0f} | spend ${d['marketing_spend']:,.0f} | "
                   f"lost {d['lost_sales_units']:,.0f} | expired {d['expired_units']:,.0f}")
    print("NO PROMO  :", f(none_kpis))
    print("FLAT 20%  :", f(base_kpis))
    print("OPTIMIZER :", f(kpis))
    print(f"sell-through: { {a: f'{b:.0%}' for a, b in kpis['sell_through'].items()} }")
    print(f"constraint violations ({len(viol)} rules broken):")
    for v in viol:
        print(f"  - {v['rule']}: {v['count']}  {v['example']}")

    out = {
        "status": status,
        "optimizer": kpis,
        "flat_20pct": base_kpis,
        "no_promo": none_kpis,
        "violations": viol,
        "hard_violations": hard_violations,
        "solution_status": plan.attrs["solution_status"],
        "clearance_slack_units": plan.attrs["clearance_slack_units"],
        "clearance_shortfalls": {
            clearance_key(tg): max(0.0, tg["min_sell_through_pct"] - kpis["sell_through"][clearance_key(tg)])
            for tg in c.get("clearance_targets", [])
        },
        "sell_through": kpis["sell_through"],
    }
    (DATA / "optimizer_results.json").write_text(json.dumps(out, indent=2, default=float))
    plan.to_csv(DATA / "optimizer_plan.csv", index=False)
    print(f"written to {DATA / 'optimizer_results.json'} and {DATA / 'optimizer_plan.csv'}")


if __name__ == "__main__":
    main()
