"""Phase 2a tests: the estimator recovers the known elasticities from the generated history, using the history only."""
from pathlib import Path

import numpy as np
import pytest

from src import elasticity
from src.config import plan_weeks
from src.demand import DemandModel


@pytest.fixture(scope="module")
def est():
    return elasticity.fit()


@pytest.fixture(scope="module")
def truth():
    return DemandModel()


def test_estimator_does_not_touch_ground_truth():
    src = Path(elasticity.__file__).read_text().lower()
    assert "gold" not in src and "from .demand" not in src


def test_own_elasticity_recovered(est, truth):
    err = est.E - truth.E
    assert (est.E < 0).all()
    assert np.abs(err).mean() < 0.15
    assert np.abs(err).max() < 0.40
    assert (np.abs(err) <= 3 * est.E_se).mean() >= 0.85      # standard errors are honest enough for Monte Carlo


def test_cross_elasticity_structure_recovered(est, truth):
    Ct = truth.C
    strong = np.abs(Ct) >= 0.25                                # substitutes and the strong complements
    assert strong.sum() >= 40
    assert (np.sign(est.C_raw[strong]) == np.sign(Ct[strong])).mean() >= 0.95
    assert np.abs(est.C_raw[strong] - Ct[strong]).mean() < 0.15
    assert (est.C != 0)[strong].mean() >= 0.80                 # screen keeps most real links
    assert ((est.C != 0) & (Ct == 0)).sum() <= 8               # and almost no invented ones


def test_relationship_table_has_both_kinds(est):
    pairs = est.pairs()
    assert {"substitute", "complement"} <= set(pairs.relation)
    row = pairs[(pairs.affected_sku == "C001") & (pairs.price_sku == "C002")]
    assert len(row) == 1 and row.relation.iloc[0] == "complement"   # shampoo and conditioner


def test_baseline_forecast_matches_world(est, truth):
    ones = np.ones((8, 48))
    errs = []
    for ws in plan_weeks():
        ratio = est.expected_units(ws, ones).sum(axis=0) / truth.expected_units(ws, ones).sum(axis=0)
        errs.append(np.abs(ratio - 1))
    errs = np.array(errs)
    assert errs.max() < 0.20
    assert errs.mean() < 0.05


def test_discount_response_matches_world(est, truth):
    ws = plan_weeks()[1]
    ones = np.ones((8, 48))
    ratio = ones.copy()
    j, k = truth.sku_idx["B001"], truth.sku_idx["B002"]
    ratio[:, j] = 0.8
    lift_est = est.expected_units(ws, ratio) / est.expected_units(ws, ones)
    lift_true = truth.expected_units(ws, ratio) / truth.expected_units(ws, ones)
    assert abs(lift_est[:, j].mean() / lift_true[:, j].mean() - 1) < 0.10    # own lift
    assert lift_est[:, k].mean() < 1 and lift_true[:, k].mean() < 1          # diet cola is cannibalised


def test_draw_perturbs_values_but_keeps_structure(est):
    d = est.draw(np.random.default_rng(0))
    assert d.E.shape == est.E.shape
    assert ((d.C != 0) == (est.C != 0)).all()
    assert not np.allclose(d.E, est.E)
