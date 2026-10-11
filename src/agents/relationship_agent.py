"""Phase 3 – Relationship Agent.

Combines the estimated cross-elasticity pairs (from src/elasticity) with basket
co-purchase affinity to identify substitute and complement groups, then asks the
LLM to name each group and write a one-sentence planning rationale.

Usage:
    from src.agents.relationship_agent import run
    groups = run()          # list[RelationshipGroup]

Run standalone:  python -m src.agents.relationship_agent
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..config import DATA
from ._llm import call_json

_SYSTEM = """You are a retail category analyst.
You will receive two tables:
1. cross_elasticity_pairs – estimated price cross-elasticities between SKUs
   (substitute > 0 means discounting one cannibalises the other;
    complement < 0 means discounting one lifts the other).
2. basket_affinity – top co-purchase pairs by lift (how often bought together vs chance).

Your task: identify meaningful product relationship groups for promotion planning.

Return ONLY a JSON array of group objects:
[
  {
    "group_id": "<short_id>",
    "name": "<human-readable group name>",
    "relation": "substitute" | "complement" | "mixed",
    "skus": ["<sku>", ...],
    "planning_note": "<one sentence: what this means for promotion planning>",
    "cannibalization_risk": "high" | "medium" | "low" | "none"
  },
  ...
]

Rules:
- Only include groups with at least 2 SKUs and clear evidence from the data.
- For substitutes: flag cannibalization_risk as high if |elasticity| >= 0.40, medium if >= 0.25.
- For complements: note the halo opportunity in planning_note.
- Merge SKUs into one group if they appear in both tables (cross-elasticity AND basket affinity).
- Do not invent relationships not supported by the data.
- Keep groups tight (2-4 SKUs); avoid mega-groups.
- IMPORTANT: B001 (FizzUp Cola) and B002 (FizzUp Diet Cola) are in the same Soda subcategory
  and are mutually exclusive in promotions — they must appear together in a substitute group.
  Their cross-elasticity may be screened out statistically but the business constraint confirms substitution.
"""


@dataclass
class RelationshipGroup:
    group_id: str
    name: str
    relation: str
    skus: list[str]
    planning_note: str
    cannibalization_risk: str


def _basket_affinity(top_n: int = 30) -> pd.DataFrame:
    """Compute top co-purchase pairs by lift from basket_items.csv."""
    baskets = pd.read_csv(DATA / "basket_items.csv")
    n_baskets = baskets.basket_id.nunique()
    sku_freq = baskets.groupby("sku").basket_id.nunique() / n_baskets

    pairs = (baskets.merge(baskets, on="basket_id", suffixes=("_a", "_b"))
             .query("sku_a < sku_b")
             .groupby(["sku_a", "sku_b"]).basket_id.nunique()
             .reset_index(name="co_count"))
    pairs["lift"] = (pairs.co_count / n_baskets) / (
        pairs.sku_a.map(sku_freq) * pairs.sku_b.map(sku_freq)
    )
    return pairs.nlargest(top_n, "lift")[["sku_a", "sku_b", "lift", "co_count"]]


def run(est=None) -> list[RelationshipGroup]:
    """Return relationship groups.  Pass a DemandEstimates object or it will be fitted."""
    if est is None:
        from ..elasticity import fit
        est = fit()

    pairs_df = est.pairs()
    affinity_df = _basket_affinity()

    pairs_str = pairs_df.head(40).to_csv(index=False)
    affinity_str = affinity_df.to_csv(index=False)

    user_msg = (
        f"cross_elasticity_pairs (top 40 by |elasticity|):\n{pairs_str}\n\n"
        f"basket_affinity (top 30 by lift):\n{affinity_str}"
    )

    raw = call_json(_SYSTEM, user_msg)
    return [RelationshipGroup(**g) for g in raw]


def main():
    groups = run()
    print(f"\n=== Relationship Agent: {len(groups)} groups ===")
    for g in groups:
        print(f"\n[{g.group_id}] {g.name}  ({g.relation}, cannibalization={g.cannibalization_risk})")
        print(f"  SKUs: {', '.join(g.skus)}")
        print(f"  Note: {g.planning_note}")


if __name__ == "__main__":
    main()
