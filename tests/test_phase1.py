"""Phase 1 tests: schema integrity, generator determinism, demand-model sanity, text inputs, baseline, validator."""
import json

import numpy as np
import pandas as pd
import pytest

from src import baseline, generate_data
from src.config import DATA, GOLD, INPUTS, plan_weeks
from src.demand import DemandModel
from src.evaluator import check_constraints, evaluate_plan, load_constraints, load_inventory

W = plan_weeks()


@pytest.fixture(scope="session", autouse=True)
def built():
    if not (GOLD / "constraints.json").exists():
        generate_data.main()
        from src import generate_inputs
        generate_inputs.main()


@pytest.fixture(scope="session")
def model():
    return DemandModel()


@pytest.fixture(scope="session")
def c():
    return load_constraints()


@pytest.fixture(scope="session")
def inv():
    return load_inventory()


def rows(sku, weeks, d, stores):
    return [(W[w], s, sku, d) for w in weeks for s in stores]


def mkplan(r):
    return pd.DataFrame(r, columns=["week_start", "store_id", "sku", "discount_pct"])


def feasible_plan(model):
    S = model.store_ids
    r = rows("C008", [0, 1, 2, 4], 0.35, S) + rows("P006", [0, 1], 0.20, S) + rows("S008", [0, 1, 2], 0.25, S)
    r += rows("D002", [0, 1], 0.15, ["S03", "S05"])
    for sku in ("C004", "C007", "H002"):
        r += rows(sku, [0, 1, 2], 0.15, S)
    return mkplan(r)


# ---------- schema ----------
def test_tables_and_keys(model):
    sales = pd.read_csv(DATA / "sales_history.csv")
    assert list(sales.columns) == ["week_start", "store_id", "sku", "sell_price", "units", "on_promo"]
    assert sales.week_start.nunique() == 104 and len(model.skus) == 48 and len(model.store_ids) == 8
    assert len(sales) == 104 * 8 * 48
    assert not sales.duplicated(["week_start", "store_id", "sku"]).any()
    assert (sales.units >= 0).all() and (sales.sell_price > 0).all() and sales.isna().sum().sum() == 0
    p = model.products
    assert p.sku.is_unique and (p.unit_cost < p.list_price).all()
    margin = 1 - p.unit_cost / p.list_price
    assert margin.between(0.15, 0.55).all()
    inv = pd.read_csv(DATA / "inventory.csv")
    assert len(inv) == 8 * 48 and not inv.duplicated(["store_id", "sku"]).any() and (inv.on_hand_units >= 0).all()
    comp = pd.read_csv(DATA / "competitor_prices.csv")
    assert set(comp.competitor) == {"MegaMart", "ValueHub", "FreshCo"} and len(comp) == 3 * 48
    assert set(pd.read_csv(DATA / "segment_category.csv").groupby("category").unit_share.sum().round(3)) == {1.0}
    b = pd.read_csv(DATA / "basket_items.csv")
    assert b.basket_id.nunique() >= 29000 and set(b.sku) <= set(model.skus)


def test_generators_deterministic(model):
    a = generate_data.gen_sales(model, np.random.default_rng(7))
    b = generate_data.gen_sales(model, np.random.default_rng(7))
    pd.testing.assert_frame_equal(a, b)


# ---------- demand model sanity ----------
def test_demand_directions(model):
    ws = W[0]
    ones = np.ones((8, 48))
    base = model.expected_units(ws, ones)
    j = model.sku_idx

    def lift(sku, d):
        r = ones.copy(); r[:, j[sku]] = 1 - d
        return model.expected_units(ws, r) / base

    assert (lift("B001", 0.2)[:, j["B001"]] > 1.3).all()          # own lift
    assert (lift("B001", 0.2)[:, j["B002"]] < 1).all()            # substitute cannibalised
    assert (lift("S005", 0.2)[:, j["S001"]] > 1).all()            # complement halo
    assert (lift("B001", 0.2)[:, j["D001"]] == pytest.approx(1.0))  # unrelated untouched
    assert model.seasonal(pd.Timestamp("2026-07-13"))[j["C008"]] > 1.5 * model.seasonal(pd.Timestamp("2026-11-30"))[j["C008"]]


def test_history_identifies_true_elasticity(model):
    """Sanity that Phase 2 can recover elasticity from the generated history (store+week fixed effects)."""
    s = pd.read_csv(DATA / "sales_history.csv")
    lp = model.products.set_index("sku").list_price
    for sku, true_e in [("B001", -2.3), ("D008", -1.1), ("H005", -1.7)]:
        d = s[s.sku == sku].copy()
        d["y"] = np.log(d.units.clip(lower=1)); d["x"] = np.log(d.sell_price / lp[sku])
        for col in ("y", "x"):
            d[col] = d[col] - d.groupby("store_id")[col].transform("mean") - d.groupby("week_start")[col].transform("mean") + d[col].mean()
        slope = (d.x * d.y).sum() / (d.x ** 2).sum()
        assert abs(slope - true_e) < 0.6, (sku, slope, true_e)


# ---------- text inputs & gold ----------
def test_text_inputs_present_and_consistent(c):
    names = ["brand_guidelines_email", "competitor_price_note", "budget_memo", "ops_inventory_note"]
    txt = {n: (INPUTS / f"{n}.txt").read_text() for n in names}
    assert all(len(t) > 400 for t in txt.values())
    assert f"${c['marketing_budget']:,}" in txt["budget_memo"]
    assert "5.49" in txt["competitor_price_note"] and "Roast Masters" in txt["brand_guidelines_email"]
    assert "9 Nov" in txt["ops_inventory_note"]
    traps = json.loads((GOLD / "extraction_traps.json").read_text())
    assert len(traps["competitor_updates"]) == 6


def test_constraints_gold_complete(c):
    for key in ["marketing_budget", "min_blended_promo_margin_pct", "min_sku_margin_pct", "max_discount_pct",
                "max_discount_by_brand", "no_promo_skus", "clearance_targets", "blackouts"]:
        assert key in c
    assert c["contingent_budget_not_to_plan_on"] == 5000 and len(c["clearance_targets"]) == 4


# ---------- evaluator ----------
def test_evaluator_inventory_conservation(model, inv, c):
    k, sold = evaluate_plan(baseline.flat_plan(model), model, inv, c)
    i = inv.set_index(["store_id", "sku"]).loc[[(s, x) for s in model.store_ids for x in model.skus]]
    cap = i.on_hand_units.to_numpy().reshape(8, 48) + 5 * i.weekly_resupply_units.to_numpy().reshape(8, 48)
    assert (sold <= cap + 1e-6).all() and (sold >= 0).all()


def test_noise_mode_mean_matches_deterministic(model, inv, c):
    plan = feasible_plan(model)
    det, _ = evaluate_plan(plan, model, inv, c)
    rng = np.random.default_rng(1)
    sims = [evaluate_plan(plan, model, inv, c, noise=True, rng=rng)[0]["revenue"] for _ in range(8)]
    assert abs(np.mean(sims) / det["revenue"] - 1) < 0.02


# ---------- baseline ----------
def test_flat20_baseline(model, inv, c):
    plan, k, viol = baseline.run_baseline(model, c)
    assert len(plan) == 5 * 8 * 48 and plan.discount_pct.eq(0.20).all()
    none_k, _ = evaluate_plan(plan.iloc[0:0], model, inv, c)
    assert k["marketing_spend"] > 2.5 * c["marketing_budget"]
    assert k["gross_margin_pct"] < none_k["gross_margin_pct"] - 0.10     # destroys margin
    assert k["revenue"] > none_k["revenue"]                                # but does lift revenue
    rules = {v["rule"] for v in viol}
    assert {"marketing_budget", "min_blended_promo_margin", "min_sku_margin", "no_promo_sku",
            "max_consecutive_weeks", "clearance:C008@all"} <= rules
    assert len(viol) >= 8


# ---------- validator ----------
def test_validator_accepts_feasible_plan(model, inv, c):
    plan = feasible_plan(model)
    k, _ = evaluate_plan(plan, model, inv, c)
    assert check_constraints(plan, k, c, model.products, model.stores) == []
    assert k["marketing_spend"] < c["marketing_budget"]


@pytest.mark.parametrize("extra,rule", [
    (lambda m: rows("D001", [0], 0.10, ["S01"]), "no_promo_sku"),
    (lambda m: rows("B001", [0], 0.30, ["S01"]), "max_discount_brand:FizzUp"),
    (lambda m: rows("D008", [1], 0.20, ["S02"]), "max_discount_brand:Roast Masters"),
    (lambda m: rows("P004", [0], 0.10, ["S07"]), "blackout:P004@West"),
    (lambda m: rows("B001", [3], 0.10, ["S01"]) + rows("B002", [3], 0.10, ["S01"]), "mutually_exclusive:B001/B002"),
    (lambda m: rows("C001", [3], 0.10, ["S01"]), "promote_together:C001/C002"),
    (lambda m: rows("D003", [0], 0.30, ["S01"]), "min_sku_margin"),
    (lambda m: rows("H001", [0, 1, 2, 3], 0.10, ["S01"]), "max_consecutive_weeks"),
    (lambda m: rows("S001", [3], 0.10, ["S01"]) + rows("S002", [3], 0.10, ["S01"]) + rows("S003", [3], 0.10, ["S01"])
     + rows("S004", [3], 0.10, ["S01"]), "max_skus_per_category_per_store_week"),
    (lambda m: rows("B003", [3], 0.36, ["S01"]), "max_discount_global"),
])
def test_validator_catches_violation(model, inv, c, extra, rule):
    plan = pd.concat([feasible_plan(model), mkplan(extra(model))], ignore_index=True)
    k, _ = evaluate_plan(plan, model, inv, c)
    assert rule in {v["rule"] for v in check_constraints(plan, k, c, model.products, model.stores)}
