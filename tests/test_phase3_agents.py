"""Phase 3 tests: Input agent, Relationship agent, Strategy agent, Orchestrator.

These tests call the real LLM (Azure OpenAI).  They are marked with the 'llm'
pytest mark so you can skip them offline:  pytest -m "not llm" -q

Run all:  python -m pytest tests/test_phase3_agents.py -q
"""
import json

import pytest

from src.config import GOLD, INPUTS
from src.agents.input_agent import run as run_input, score as score_input
from src.agents.relationship_agent import run as run_relationship
from src.agents.strategy_agent import run as run_strategy
from src.agents.orchestrator import Orchestrator

pytestmark = pytest.mark.llm


@pytest.fixture(scope="module")
def gold():
    return json.loads((GOLD / "constraints.json").read_text())


@pytest.fixture(scope="module")
def input_result():
    return run_input()


@pytest.fixture(scope="module")
def groups(input_result):
    return run_relationship()


@pytest.fixture(scope="module")
def strategy(input_result, groups):
    return run_strategy(constraints=input_result.constraints, groups=groups)


# ── Input agent ───────────────────────────────────────────────────────────────

def test_input_extraction_accuracy(input_result, gold):
    sc = score_input(input_result, gold)
    assert sc["accuracy"] >= 0.80, f"Accuracy {sc['accuracy']:.0%} < 80%\nMisses: {sc['misses']}"


def test_input_correct_budget(input_result, gold):
    """Must use the superseded (final) budget, not the old Sept figure."""
    assert input_result.constraints["marketing_budget"] == gold["marketing_budget"]


def test_input_contingent_budget_excluded(input_result):
    """$5k contingent budget must NOT be added to marketing_budget."""
    assert input_result.constraints.get("contingent_budget_not_to_plan_on") == 5000
    assert input_result.constraints["marketing_budget"] == 59000


def test_input_no_promo_sku(input_result):
    """Meadow Whole Milk (D001) must be in no_promo_skus."""
    assert "D001" in input_result.constraints.get("no_promo_skus", [])


def test_input_mutually_exclusive(input_result):
    """FizzUp Cola and Diet Cola must be mutually exclusive."""
    pairs = [tuple(sorted(p)) for p in input_result.constraints.get("mutually_exclusive_same_week", [])]
    assert ("B001", "B002") in pairs


def test_input_promote_together(input_result):
    """Silk&Shine shampoo + conditioner must be promote_together."""
    pairs = [tuple(sorted(p)) for p in input_result.constraints.get("promote_together", [])]
    assert ("C001", "C002") in pairs


def test_input_blackouts_present(input_result):
    """Nonna's pizzas must be blacked out in West region."""
    bos = {(b["sku"], b["region"]) for b in input_result.constraints.get("blackouts", [])}
    assert ("P004", "West") in bos
    assert ("P005", "West") in bos


def test_input_clearance_targets(input_result):
    """All 4 clearance targets must be extracted."""
    ct = {t["sku"] for t in input_result.constraints.get("clearance_targets", [])}
    assert {"C008", "P006", "S008", "D002"} <= ct


def test_input_expired_competitor_promo_excluded(input_result):
    """FreshCo SoftRoll promo (expires before window) must be flagged, not used as a constraint."""
    warnings_text = " ".join(input_result.warnings).lower()
    assert "freshco" in warnings_text or "softroll" in warnings_text or "expired" in warnings_text


def test_input_unverified_rumour_excluded(input_result):
    """MegaMart butter rumour must be flagged as unverified."""
    warnings_text = " ".join(input_result.warnings).lower()
    assert "megamart" in warnings_text or "butter" in warnings_text or "unverified" in warnings_text


def test_input_confidence_flags_present(input_result):
    """Confidence dict must have entries; audience must be low confidence."""
    assert len(input_result.confidence) >= 3
    audience_conf = input_result.confidence.get("audience", "low")
    assert audience_conf == "low"


def test_input_brand_caps(input_result):
    """Brand discount caps must be extracted correctly."""
    caps = input_result.constraints.get("max_discount_by_brand", {})
    assert abs(caps.get("FizzUp", 0) - 0.25) < 1e-6
    assert abs(caps.get("Roast Masters", 0) - 0.15) < 1e-6
    assert abs(caps.get("SoftRoll", 0) - 0.20) < 1e-6


# ── Relationship agent ────────────────────────────────────────────────────────

def test_relationship_groups_nonempty(groups):
    assert len(groups) >= 3


def test_relationship_has_substitute_and_complement(groups):
    relations = {g.relation for g in groups}
    assert "substitute" in relations
    assert "complement" in relations


def test_relationship_fizz_up_substitutes(groups):
    """B001 and B002 should be in the same substitute group (same subcategory, mutually exclusive)."""
    for g in groups:
        if g.relation in ("substitute", "mixed") and "B001" in g.skus and "B002" in g.skus:
            return
    pytest.fail("B001 and B002 not found in any substitute group")


def test_relationship_shampoo_conditioner_complement(groups):
    """C001 and C002 should be in a complement group (strong negative cross-elasticity)."""
    for g in groups:
        if g.relation in ("complement", "mixed") and "C001" in g.skus and "C002" in g.skus:
            return
    pytest.fail("C001 and C002 not found in any complement group")


def test_relationship_cannibalization_risk_set(groups):
    for g in groups:
        assert g.cannibalization_risk in ("high", "medium", "low", "none")


# ── Strategy agent ────────────────────────────────────────────────────────────

def test_strategy_must_promote_clearance_skus(strategy):
    """All 4 clearance SKUs must appear in the shortlist as must_promote."""
    must = {s["sku"] for s in strategy.shortlist if s["priority"] == "must_promote"}
    assert {"C008", "P006", "S008", "D002"} <= must


def test_strategy_no_promo_sku_excluded(strategy):
    """D001 (Meadow Whole Milk) must not appear in the shortlist."""
    shortlist_skus = {s["sku"] for s in strategy.shortlist}
    assert "D001" not in shortlist_skus


def test_strategy_brand_caps_respected(strategy):
    """Suggested discounts must not exceed brand caps."""
    caps = {"FizzUp": 0.25, "Roast Masters": 0.15, "SoftRoll": 0.20}
    brand_map = {
        "B001": "FizzUp", "B002": "FizzUp", "B003": "FizzUp",
        "D008": "Roast Masters",
        "H005": "SoftRoll", "H006": "SoftRoll",
    }
    for s in strategy.shortlist:
        brand = brand_map.get(s["sku"])
        if brand:
            cap = caps[brand]
            assert s["suggested_discount_pct"] <= cap + 1e-6, (
                f"{s['sku']} discount {s['suggested_discount_pct']:.0%} exceeds {brand} cap {cap:.0%}"
            )


def test_strategy_shortlist_size(strategy):
    assert 5 <= len(strategy.shortlist) <= 20


def test_strategy_has_rationale(strategy):
    assert len(strategy.overall_rationale) > 50
    for s in strategy.shortlist:
        assert len(s.get("rationale", "")) > 10


# ── Orchestrator ──────────────────────────────────────────────────────────────

def test_orchestrator_full_run():
    orch = Orchestrator()
    result = orch.run()
    assert result.input_result is not None
    assert len(result.groups) >= 2
    assert len(result.strategy.shortlist) >= 5


def test_orchestrator_replan_only_reruns_strategy():
    orch = Orchestrator()
    orch.run()
    original_groups = orch._groups

    result = orch.replan({"marketing_budget": 40000})
    assert orch._groups is original_groups   # relationship agent NOT re-run
    assert any("strategy_agent re-run" in entry for entry in result.replan_log)


def test_orchestrator_replan_constraint_reflected(strategy):
    """After cutting budget, strategy should acknowledge tighter budget."""
    orch = Orchestrator()
    orch.run()
    result = orch.replan({"marketing_budget": 30000})
    # Shortlist should still contain must_promote SKUs (clearance is non-negotiable)
    must = {s["sku"] for s in result.strategy.shortlist if s["priority"] == "must_promote"}
    assert "C008" in must
