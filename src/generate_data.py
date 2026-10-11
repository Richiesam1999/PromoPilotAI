"""Phase 1 synthetic data generator.  Run:  python -m src.generate_data

Writes (data/):  products, stores, segments, segment_category, store_segment_mix, holidays, holiday_lifts,
                 sales_history, basket_items, inventory, competitor_prices
Writes (data/gold/): true_demand_params, true_cross_elasticity, region_category_index
"""
import numpy as np
import pandas as pd

from . import catalog
from .config import (COMPETITOR_SNAPSHOT, DATA, GOLD, HISTORY_LAST_WEEK, HISTORY_WEEKS, INPUTS, PLAN_START,
                     SEED, plan_weeks)
from .demand import DemandModel

N_BASKETS = 30000
PROMO_PROB = 0.15
PROMO_DEPTHS = [0.10, 0.15, 0.20, 0.25, 0.30]

# inventory scenario knobs (these create the clearance / expiry / supply-outage stories)
CLEARANCE_COVER = {"C008": (15, 21), "P006": (7, 10), "S008": (8, 11)}   # weeks of cover, no resupply
EXPIRY_STORES = {"D002": ["S03", "S05"]}     # batch expiring; cover (weeks), days to expiry
EXPIRY_COVER, EXPIRY_DAYS = 3.0, 14
OUTAGE = {"skus": ["P004", "P005"], "stores": ["S07", "S08"], "cover": 0.9, "resupply_start": "2026-11-09"}


def write_static(rng):
    DATA.mkdir(exist_ok=True), GOLD.mkdir(exist_ok=True), INPUTS.mkdir(exist_ok=True)
    prod = catalog.products_df()
    prod.to_csv(DATA / "products.csv", index=False)
    catalog.stores_df().to_csv(DATA / "stores.csv", index=False)
    seg, seg_cat, mix = catalog.segment_frames(rng)
    seg.to_csv(DATA / "segments.csv", index=False)
    seg_cat.to_csv(DATA / "segment_category.csv", index=False)
    mix.to_csv(DATA / "store_segment_mix.csv", index=False)
    catalog.holidays_df().to_csv(DATA / "holidays.csv", index=False)
    catalog.holiday_lifts_df().to_csv(DATA / "holiday_lifts.csv", index=False)
    catalog.true_demand_params_df().to_csv(GOLD / "true_demand_params.csv", index=False)
    catalog.true_cross_matrix(prod).to_csv(GOLD / "true_cross_elasticity.csv")
    pd.DataFrame([(r, c, v) for r, d in catalog.REGION_CATEGORY_INDEX.items() for c, v in d.items()],
                 columns=["region", "category", "index"]).to_csv(GOLD / "region_category_index.csv", index=False)


def gen_sales(model, rng):
    weeks = pd.date_range(end=HISTORY_LAST_WEEK, periods=HISTORY_WEEKS, freq="W-MON")
    S, K = len(model.store_ids), len(model.skus)
    lp = model.products.list_price.to_numpy()
    rows = []
    for ws in weeks:
        chain = np.where(rng.random(K) < PROMO_PROB, 1 - rng.choice(PROMO_DEPTHS, K), 1.0)
        part = rng.random((S, K)) < 0.9                      # not every store runs the chain promo
        ratio = np.where(part, chain[None, :], 1.0)
        price = np.round(lp[None, :] * ratio, 2)
        ratio = price / lp[None, :]
        mean = model.expected_units(ws, ratio) * rng.lognormal(0, 0.08, (S, K))
        units = rng.poisson(mean)
        rows.append(pd.DataFrame({
            "week_start": ws, "store_id": np.repeat(model.store_ids, K), "sku": np.tile(model.skus, S),
            "sell_price": price.ravel(), "units": units.ravel(), "on_promo": (ratio.ravel() < 0.97).astype(int)}))
    return pd.concat(rows, ignore_index=True)


def gen_baskets(model, rng):
    """Synthetic baskets: complements co-purchased more often, substitutes rarely together."""
    K = len(model.skus)
    pop = model.base / model.base.sum()
    sub = model.products.subcategory.to_numpy()
    comp = {}
    for j in range(K):
        for k in range(K):
            if model.C[j, k] < -0.2:      # k's presence pulls in j
                comp.setdefault(k, []).append((j, -model.C[j, k]))
    rows = []
    store_p = model.size / model.size.sum()
    for b in range(N_BASKETS):
        st = model.store_ids[rng.choice(len(model.store_ids), p=store_p)]
        n = int(np.clip(rng.poisson(3.5) + 2, 2, 9))
        items = set()
        for k in rng.choice(K, n, p=pop):
            if any(sub[k] == sub[i] for i in items) and rng.random() < 0.9:
                continue                                      # one item per subcategory, mostly
            items.add(k)
            for j, strength in comp.get(k, []):
                if rng.random() < strength:
                    items.add(j)
        rows += [(b, st, model.skus[i]) for i in items]
    return pd.DataFrame(rows, columns=["basket_id", "store_id", "sku"])


def gen_inventory(model, rng):
    weeks = plan_weeks()
    r_avg = np.mean([model.baseline_units(w) for w in weeks], axis=0)    # (S,K) normal run-rate in window
    S, K = r_avg.shape
    cover = rng.lognormal(np.log(2.5), 0.2, (S, K))
    resupply = np.round(r_avg)
    start = np.full((S, K), np.nan, dtype=object)
    expiry = np.full((S, K), np.nan)
    for sku, (lo, hi) in CLEARANCE_COVER.items():
        k = model.sku_idx[sku]
        cover[:, k] = rng.uniform(lo, hi, S)
        resupply[:, k] = 0
    for sku, stores in EXPIRY_STORES.items():
        for st in stores:
            s, k = model.store_idx[st], model.sku_idx[sku]
            cover[s, k], resupply[s, k], expiry[s, k] = EXPIRY_COVER, 0, EXPIRY_DAYS
    for sku in OUTAGE["skus"]:
        for st in OUTAGE["stores"]:
            s, k = model.store_idx[st], model.sku_idx[sku]
            cover[s, k] = OUTAGE["cover"]
            start[s, k] = OUTAGE["resupply_start"]
    on_hand = np.round(r_avg * cover)
    return pd.DataFrame({
        "store_id": np.repeat(model.store_ids, K), "sku": np.tile(model.skus, S),
        "on_hand_units": on_hand.ravel().astype(int), "weekly_resupply_units": resupply.ravel().astype(int),
        "resupply_start_date": start.ravel(), "days_to_expiry": expiry.ravel()})


COMPETITORS = {"MegaMart": (0.97, 0.05), "ValueHub": (0.91, 0.06), "FreshCo": (1.04, 0.05)}
STALE_OVERRIDES = {("ValueHub", "B001"): 6.29, ("ValueHub", "P004"): 5.79, ("ValueHub", "P005"): 5.79,
                   ("ValueHub", "C008"): 8.99, ("FreshCo", "H005"): 8.79}


def gen_competitors(model, rng):
    rows = []
    for comp, (mu, sd) in COMPETITORS.items():
        for sku, lp in zip(model.skus, model.products.list_price):
            p = lp * rng.normal(mu, sd)
            p = max(0.49, round(p * 10) / 10 - 0.01)
            p = STALE_OVERRIDES.get((comp, sku), p)
            rows.append((COMPETITOR_SNAPSHOT, comp, sku, round(p, 2), int(p <= 0.88 * lp)))
    return pd.DataFrame(rows, columns=["as_of_date", "competitor", "sku", "price", "on_promo"])


def main():
    rng = np.random.default_rng(SEED)
    write_static(rng)
    model = DemandModel()
    gen_sales(model, rng).to_csv(DATA / "sales_history.csv", index=False)
    gen_baskets(model, rng).to_csv(DATA / "basket_items.csv", index=False)
    gen_inventory(model, rng).to_csv(DATA / "inventory.csv", index=False)
    gen_competitors(model, rng).to_csv(DATA / "competitor_prices.csv", index=False)
    print("Data written to", DATA)


if __name__ == "__main__":
    main()
