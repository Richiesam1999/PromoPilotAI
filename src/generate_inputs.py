"""Writes the constraints (ground truth), the 4 messy text inputs and the extraction gold. Run: python -m src.generate_inputs

The text files are what the Phase-3 Input Agent will read.  gold/constraints.json is what it SHOULD extract.
Numbers in the texts are filled from the generated data so text, tables and constraints never disagree.
"""
import json

import pandas as pd

from .config import AS_OF, DATA, EXEC_COST, GOLD, INPUTS, PLAN_END, PLAN_START, PLAN_WEEKS
from .demand import DemandModel
from .evaluator import evaluate_plan, load_inventory
from .baseline import flat_plan

CLEARANCE = [  # sku, stores, min sell-through  (feasible within margin floor; flat-20% misses the sunscreen one)
    ("C008", "all", 0.55), ("P006", "all", 0.70), ("S008", "all", 0.65), ("D002", ["S03", "S05"], 0.80)]


def r50(x):
    return int(round(x / 100.0) * 100)


def build_constraints(model, inv):
    base_k, _ = evaluate_plan(flat_plan(model), model, inv)
    budget = int(round(0.30 * base_k["marketing_spend"], -3))
    return {
        "planning_window": {"start": str(PLAN_START.date()), "end": str(PLAN_END.date()), "weeks": PLAN_WEEKS},
        "marketing_budget": budget,
        "execution_cost_per_store_sku_week": EXEC_COST,
        "min_blended_promo_margin_pct": 0.21,
        "min_sku_margin_pct": 0.12,
        "max_discount_pct": 0.35,
        "max_discount_by_brand": {"FizzUp": 0.25, "Roast Masters": 0.15, "SoftRoll": 0.20},
        "max_discount_private_label": 0.30,
        "no_promo_skus": ["D001"],
        "max_consecutive_promo_weeks": 3,
        "max_promoted_skus_per_category_per_store_week": 3,
        "mutually_exclusive_same_week": [["B001", "B002"]],
        "promote_together": [["C001", "C002"]],
        "blackouts": [{"sku": s, "region": "West", "until": "2026-11-09"} for s in ("P004", "P005")],
        "clearance_targets": [{"sku": s, "stores": st, "min_sell_through_pct": v} for s, st, v in CLEARANCE],
        "contingent_budget_not_to_plan_on": 5000,
    }, base_k


def write_texts(c, model, inv):
    inv = inv.set_index(["store_id", "sku"]).on_hand_units
    tot = lambda sku, stores=None: int(sum(inv[(s, sku)] for s in (stores or model.store_ids)))
    sun, ice, trail, yog = tot("C008"), tot("P006"), tot("S008"), tot("D002", ["S03", "S05"])
    pct = {(t["sku"]): int(t["min_sell_through_pct"] * 100) for t in c["clearance_targets"]}
    B = c["marketing_budget"]

    (INPUTS / "brand_guidelines_email.txt").write_text(f"""From: Priya Nair <priya.nair@retailco.example>
To: promo-planning@retailco.example
Date: Thu, 8 Oct 2026 17:42
Subject: RE: RE: Q4 promos - brand stuff to keep in mind (pls read!!)

hi team,

quick list before you lock the Q4 plan, sorry for the formatting - typing on my phone

- FizzUp: fine to promo but NOT deeper than 25% off. and never run the diet cola at the same time as the regular
  cola, we dont want to cannibalise ourselves
- Roast Masters coffee is premium, anything over 15% off damages the brand. 15 max, thanks. (this one came from the CMO
  so its not negotiable)
- SoftRoll (paper towels + TP): max 20% off and not more than 3 weeks in a row, trade partner contract
- Meadow whole milk is our traffic anchor + margin is razor thin. please DO NOT put it on promo. (yogurt / cheese ok)
- Store brand stuff we can go to 30% (margin is healthy)  but nothing below 12% margin pls, finance rule
- Silk&Shine shampoo and conditioner - vendor wants them promoted together or not at all
- across the board: max 3 items per category on promo in any one week in a store (signage/space), 3 weeks max in a
  row on any item, and nothing deeper than 35% anywhere, it looks desperate

thx!
Priya
""")
    (INPUTS / "competitor_price_note.txt").write_text("""Field notes - Fri 9 Oct (Marcus, regional pricing)

walked ValueHub (Dallas + Atlanta) and FreshCo (Boston) today. rough notes:

* ValueHub has the big cola 12pk (the FizzUp one) at 5.49 now, was about 6.29 last week. looks like a proper price
  cut not just a flyer. didnt check the diet one
* Nonna's pizzas at ValueHub = 4.99 both flavours (pepperoni + 4 cheese), sign says "thru Nov 1" - rumour they will
  extend it but nobody confirmed
* FreshCo: SoftRoll paper towels 7.99 til sunday only, back to normal after
* MegaMart - didnt get there. someone in the Dallas store said they are undercutting on butter, I did NOT see it
  myself so dont use that
* ValueHub also has sunscreen (SPF50 8oz) down to about 6.99, probably the same overstock problem we have lol

will do Seattle + LA next week
""")
    (INPUTS / "budget_memo.txt").write_text(f"""FW: Q4 promo funding - FINAL (supersedes Sept version)

Team,

Funding for the {PLAN_START:%b} {PLAN_START.day} - {PLAN_END:%b} {PLAN_END.day} promo window is ${B:,} all-in. That is discount
cost PLUS in-store execution (about ${EXEC_COST:.0f} per store/SKU/week for tags + signage). The ${B + 12000:,} figure
from the Sept mail is dead, please ignore it.

There MAY be another ~$5k if the CFO signs off after the Nov board call - do NOT plan on it.

Margin guardrails: blended gross margin on promoted sales has to stay at or above 21%, and no single item can go
below a 12% margin after discount.

Thanksgiving week (w/c Nov 23) is the one that matters most for the board numbers.

Questions -> reply here.
Dan (FP&A)
""")
    (INPUTS / "ops_inventory_note.txt").write_text(f"""[Sam - supply chain]  9 Oct  15:10

heads up for promo planning:
- sunscreen (SunShield SPF50) - we are sitting on roughly {r50(sun):,} units chain-wide and no more POs coming. want it gone
  by end of Nov, ideally at least {pct['C008']}% of it
- vanilla ice cream 48oz also heavy (~{r50(ice):,} units), same story, at least {pct['P006']}% please
- Trailblazer trail mix ~{r50(trail):,} units, {pct['S008']}% would be nice
- Greek yogurt at Atlanta + Chicago ONLY: one batch dated to 9 Nov, ~{r50(yog):,} units between the two stores. needs to
  move ({pct['D002']}%+) before that date or we write it off
- WEST (Seattle + LA) frozen: freezer line outage. Nonna's pizzas wont get resupplied until the 9 Nov delivery and stock
  on hand is thin (<1 wk). dont promote pizza in the West before then pls
- everything else is on normal weekly replenishment - we can keep up with normal run rate only, not 2x
""")
    traps = {
        "competitor_updates": [
            {"competitor": "ValueHub", "sku": "B001", "price": 5.49, "valid_until": None, "should_use": True},
            {"competitor": "ValueHub", "sku": "P004", "price": 4.99, "valid_until": "2026-11-01", "should_use": True},
            {"competitor": "ValueHub", "sku": "P005", "price": 4.99, "valid_until": "2026-11-01", "should_use": True},
            {"competitor": "ValueHub", "sku": "C008", "price": 6.99, "valid_until": None, "should_use": True},
            {"competitor": "FreshCo", "sku": "H005", "price": 7.99, "valid_until": "2026-10-11", "should_use": False,
             "why": "expires before promo window starts"},
            {"competitor": "MegaMart", "sku": "D004", "price": None, "valid_until": None, "should_use": False,
             "why": "unverified hearsay"}],
        "traps": ["superseded_budget_in_memo", "contingent_5k_not_in_budget", "unverified_megamart_butter_rumour",
                  "freshco_promo_expires_before_window", "diet_cola_not_checked_by_field_rep",
                  "informal_product_names (the big cola, 4 cheese)", "'3 weeks in a row' + '3 items per category' limits"],
        "missing_on_purpose": "no explicit target audience / segment guidance anywhere -> agent must infer or flag low confidence",
    }
    (GOLD / "extraction_traps.json").write_text(json.dumps(traps, indent=2))


def main():
    model = DemandModel()
    inv = load_inventory()
    c, base_k = build_constraints(model, inv)
    (GOLD / "constraints.json").write_text(json.dumps(c, indent=2))
    write_texts(c, model, inv)
    print(f"constraints.json written: budget ${c['marketing_budget']:,} (flat-20% baseline would spend ${base_k['marketing_spend']:,.0f})")


if __name__ == "__main__":
    main()
