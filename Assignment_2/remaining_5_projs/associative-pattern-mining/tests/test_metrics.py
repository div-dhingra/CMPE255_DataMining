"""Tests for association rule interestingness metrics and mathematical formulas."""

import math
from backend.src.core.metrics import compute_rule_metrics


def test_perfect_association():
    # Both items always occur together: P(A) = 0.5, P(C) = 0.5, P(AUC) = 0.5
    m = compute_rule_metrics(supp_auc=0.5, supp_a=0.5, supp_c=0.5)

    assert m.support == 0.5
    assert m.confidence == 1.0
    assert m.lift == 2.0
    assert math.isinf(m.conviction) or m.conviction > 1000
    assert m.leverage == 0.25
    assert m.zhangs_metric == 1.0
    assert m.kulczynski == 1.0
    assert m.imbalance_ratio == 0.0


def test_statistical_independence():
    # Items are independent: P(A) = 0.5, P(C) = 0.4, P(AUC) = 0.5 * 0.4 = 0.20
    m = compute_rule_metrics(supp_auc=0.20, supp_a=0.50, supp_c=0.40)

    assert m.support == 0.20
    assert m.confidence == 0.40
    assert round(m.lift, 4) == 1.0
    assert round(m.conviction, 4) == 1.0
    assert abs(m.leverage) < 1e-9
    assert abs(m.zhangs_metric) < 1e-9
    assert round(m.kulczynski, 4) == 0.45


def test_negative_association():
    # Negative correlation (substitute goods): P(A) = 0.5, P(C) = 0.5, P(AUC) = 0.05
    m = compute_rule_metrics(supp_auc=0.05, supp_a=0.5, supp_c=0.5)

    assert m.confidence == 0.10
    assert m.lift < 1.0
    assert m.leverage < 0.0
    assert m.zhangs_metric < 0.0
    assert m.kulczynski < 0.5


def test_imbalance_ratio_behavior():
    # High frequency imbalance: A is a staple (supp=0.8), C is rare (supp=0.1), P(AUC)=0.08
    m = compute_rule_metrics(supp_auc=0.08, supp_a=0.80, supp_c=0.10)

    # IR = |0.8 - 0.1| / (0.8 + 0.1 - 0.08) = 0.7 / 0.82 ≈ 0.8536
    assert round(m.imbalance_ratio, 3) == 0.854
    assert 0.0 <= m.imbalance_ratio <= 1.0


def test_metric_serialization_safety():
    # Ensure to_dict safely handles infinite convictions without crashing
    m = compute_rule_metrics(supp_auc=0.4, supp_a=0.4, supp_c=0.4)
    d = m.to_dict()

    assert isinstance(d["conviction"], float)
    assert not math.isinf(d["conviction"])
    assert d["confidence"] == 1.0
