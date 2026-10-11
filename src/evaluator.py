"""Plan evaluator: pushes a promo plan through the ground-truth demand model and inventory dynamics,
and checks it against the hard constraints.  Reused by baseline, optimizer tests and Monte Carlo (Phase 4).

Plan format (DataFrame): week_start, store_id, sku, discount_pct   (rows only where discount > 0)
"""
import json

import numpy as np
import pandas as pd

from .config import DATA, EXEC_COST, GOLD, plan_weeks
from .demand import DemandModel


def load_inventory():
    return pd.read_csv(DATA / "inventory.csv", parse_dates=["resupply_start_date"])


def load_constraints():
    return json.loads((GOLD / "constraints.json").read_text())


def plan_to_array(plan, model, weeks):
    D = np.zeros((len(weeks), len(model.store_ids), len(model.skus)))
    if len(plan):
        wi = plan.week_start.map({w: i for i, w in enumerate(weeks)}).to_numpy().astype(int)
        si = plan.store_id.map(model.store_idx).to_numpy().astype(int)
        ki = plan.sku.map(model.sku_idx).to_numpy().astype(int)
        D[wi, si, ki] = plan.discount_pct.to_numpy()
    return D


def _prep_inv(inv, model):
    """Reindex inventory once; returns (stock, resup, resupply_on_mask, initial, expiry, uc_2d)."""
    S, K = len(model.store_ids), len(model.skus)
    inv_r = inv.set_index(["store_id", "sku"]).loc[[(s, k) for s in model.store_ids for k in model.skus]]
    stock = inv_r.on_hand_units.to_numpy(float).reshape(S, K)
    resup = inv_r.weekly_resupply_units.to_numpy(float).reshape(S, K)
    rstart = pd.to_datetime(inv_r.resupply_start_date).to_numpy().reshape(S, K)
    exp_days = inv_r.days_to_expiry.to_numpy(float).reshape(S, K)
    weeks = plan_weeks()
    expiry = (weeks[0] + pd.to_timedelta(exp_days.ravel(), unit="D")).to_numpy().reshape(S, K)
    resupply_masks = np.stack(
        [pd.isna(rstart) | (rstart <= np.datetime64(ws)) for ws in weeks]
    )  # (W, S, K)
    exp_masks = np.stack(
        [~pd.isna(expiry) & (expiry <= np.datetime64(ws + pd.Timedelta(days=7))) for ws in weeks]
    )  # (W, S, K)
    return stock, resup, resupply_masks, stock.copy(), exp_masks


def evaluate_plan(plan, model=None, inv=None, constraints=None, noise=False, rng=None,
                  elasticity_scale=1.0, _inv_cache=None):
    """Returns (kpis, sold) where sold is (store, sku) units sold over the window.
    noise=True draws Poisson demand (Monte Carlo); elasticity_scale perturbs own-elasticity (Phase 4).
    Pass _inv_cache=_prep_inv(inv, model) to avoid reindexing on every Monte Carlo call.
    """
    model = model or DemandModel()
    inv = inv if inv is not None else load_inventory()
    weeks = plan_weeks()
    S, K = len(model.store_ids), len(model.skus)
    D = plan_to_array(plan, model, weeks)
    lp = model.products.list_price.to_numpy()[None, :]
    uc = model.products.unit_cost.to_numpy()[None, :]

    if _inv_cache is not None:
        stock0, resup, resupply_masks, initial, exp_masks = _inv_cache
    else:
        stock0, resup, resupply_masks, initial, exp_masks = _prep_inv(inv, model)
    stock = stock0.copy()

    rng = rng or np.random.default_rng(0)
    orig_E = model.E.copy()
    model.E = orig_E * elasticity_scale
    t = dict(rev=0.0, cogs=0.0, units=0.0, lost=0.0, exp_units=0.0, exp_cost=0.0, disc=0.0,
             prev=0.0, pcogs=0.0, pweeks=0)
    sold_total = np.zeros((S, K))
    try:
        for w, ws in enumerate(weeks):
            ratio = 1 - D[w]
            mean = model.expected_units(ws, ratio)
            demand = rng.poisson(mean).astype(float) if noise else mean
            avail = stock + np.where(resupply_masks[w], resup, 0.0)
            sold = np.minimum(demand, avail)
            rev, cogs = sold * lp * ratio, sold * uc
            promo = D[w] > 0
            t["rev"] += rev.sum(); t["cogs"] += cogs.sum(); t["units"] += sold.sum()
            t["lost"] += (demand - sold).sum(); t["disc"] += (sold * lp * D[w]).sum()
            t["prev"] += rev[promo].sum(); t["pcogs"] += cogs[promo].sum(); t["pweeks"] += int(promo.sum())
            sold_total += sold
            stock = avail - sold
            exp_mask = exp_masks[w]
            t["exp_units"] += stock[exp_mask].sum()
            t["exp_cost"] += (stock * uc)[exp_mask].sum()
            stock = np.where(exp_mask, 0.0, stock)
    finally:
        model.E = orig_E

    k = {"revenue": t["rev"], "gross_margin": t["rev"] - t["cogs"], "units_sold": t["units"],
         "lost_sales_units": t["lost"], "expired_units": t["exp_units"], "writeoff_cost": t["exp_cost"],
         "discount_investment": t["disc"], "promoted_store_sku_weeks": t["pweeks"],
         "execution_cost": t["pweeks"] * EXEC_COST}
    k["marketing_spend"] = k["discount_investment"] + k["execution_cost"]
    k["gross_margin_pct"] = k["gross_margin"] / k["revenue"]
    k["promo_margin_pct"] = (t["prev"] - t["pcogs"]) / t["prev"] if t["prev"] > 0 else None
    k["net_contribution"] = k["gross_margin"] - k["execution_cost"] - k["writeoff_cost"]
    k["sell_through"] = {}
    for tg in (constraints or {}).get("clearance_targets", []):
        sel = list(range(S)) if tg["stores"] == "all" else [model.store_idx[x] for x in tg["stores"]]
        j = model.sku_idx[tg["sku"]]
        k["sell_through"][clearance_key(tg)] = float(sold_total[sel, j].sum() / max(initial[sel, j].sum(), 1))
    return k, sold_total


def clearance_key(tg):
    return f'{tg["sku"]}@' + ("all" if tg["stores"] == "all" else "+".join(tg["stores"]))


def _max_run(idx):
    run = best = 1
    for a, b in zip(idx[:-1], idx[1:]):
        run = run + 1 if b == a + 1 else 1
        best = max(best, run)
    return best


def check_constraints(plan, kpis, c, products, stores):
    """Deterministic validator (Phase 1 version). Returns list of {rule, count, example}. Empty list == feasible."""
    out = []

    def add(rule, count, example=""):
        if count:
            out.append({"rule": rule, "count": int(count), "example": example})

    weeks = plan_weeks()
    widx = {w: i for i, w in enumerate(weeks)}
    p = plan[plan.discount_pct > 0].merge(products[["sku", "brand", "category", "list_price", "unit_cost",
                                                    "is_private_label"]], on="sku").merge(
        stores[["store_id", "region"]], on="store_id")
    add("marketing_budget", kpis["marketing_spend"] > c["marketing_budget"],
        f'spend ${kpis["marketing_spend"]:,.0f} > budget ${c["marketing_budget"]:,.0f}')
    pm = kpis["promo_margin_pct"]
    add("min_blended_promo_margin", pm is not None and pm < c["min_blended_promo_margin_pct"] - 1e-9,
        f"promo margin {pm:.1%} < {c['min_blended_promo_margin_pct']:.0%}" if pm is not None else "")
    if len(p):
        price = p.list_price * (1 - p.discount_pct)
        bad = p[((price - p.unit_cost) / price) < c["min_sku_margin_pct"] - 1e-9]
        add("min_sku_margin", len(bad), f"e.g. {bad.sku.iloc[0]} at {bad.discount_pct.iloc[0]:.0%}" if len(bad) else "")
        bad = p[p.discount_pct > c["max_discount_pct"] + 1e-9]
        add("max_discount_global", len(bad))
        bad = p[(p.is_private_label == 1) & (p.discount_pct > c["max_discount_private_label"] + 1e-9)]
        add("max_discount_private_label", len(bad))
        for brand, cap in c["max_discount_by_brand"].items():
            bad = p[(p.brand == brand) & (p.discount_pct > cap + 1e-9)]
            add(f"max_discount_brand:{brand}", len(bad), f"cap {cap:.0%}")
        bad = p[p.sku.isin(c["no_promo_skus"])]
        add("no_promo_sku", len(bad), ", ".join(sorted(bad.sku.unique())))
        # consecutive weeks (per store, sku)
        viol = 0
        for _, g in p.groupby(["store_id", "sku"]):
            if _max_run(np.sort(g.week_start.map(widx).unique())) > c["max_consecutive_promo_weeks"]:
                viol += 1
        add("max_consecutive_weeks", viol, f"limit {c['max_consecutive_promo_weeks']}")
        # promoted SKUs per category per store-week
        n = p.groupby(["week_start", "store_id", "category"]).sku.nunique()
        add("max_skus_per_category_per_store_week", (n > c["max_promoted_skus_per_category_per_store_week"]).sum(),
            f"limit {c['max_promoted_skus_per_category_per_store_week']}")
        for a, b in c["mutually_exclusive_same_week"]:
            g = p[p.sku.isin([a, b])].groupby(["week_start", "store_id"]).sku.nunique()
            add(f"mutually_exclusive:{a}/{b}", (g > 1).sum())
        for a, b in c["promote_together"]:
            g = p[p.sku.isin([a, b])].groupby(["week_start", "store_id"]).sku.nunique()
            add(f"promote_together:{a}/{b}", (g == 1).sum())
        for bo in c["blackouts"]:
            bad = p[(p.sku == bo["sku"]) & (p.region == bo["region"]) & (p.week_start < pd.Timestamp(bo["until"]))]
            add(f"blackout:{bo['sku']}@{bo['region']}", len(bad), f"until {bo['until']}")
    for tg in c["clearance_targets"]:
        key = clearance_key(tg)
        got = kpis["sell_through"].get(key)
        add(f"clearance:{key}", got is not None and got < tg["min_sell_through_pct"] - 1e-9,
            f"sell-through {got:.0%} < {tg['min_sell_through_pct']:.0%}" if got is not None else "")
    return out
