"""Comprehensive association rule interestingness metrics and mathematical formulation."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
import math


@dataclass(frozen=True)
class RuleMetrics:
    """Encapsulates all 7 rule interestingness metrics plus Imbalance Ratio."""
    support: float          # P(A U C)
    antecedent_support: float # P(A)
    consequent_support: float # P(C)
    confidence: float       # P(A U C) / P(A)
    lift: float             # P(A U C) / (P(A) * P(C))
    conviction: float       # (1 - P(C)) / (1 - conf)
    leverage: float         # P(A U C) - P(A) * P(C)
    zhangs_metric: float    # Normalized association [-1, 1]
    kulczynski: float       # Null-invariant arithmetic mean of directional confidences [0, 1]
    imbalance_ratio: float  # Null-invariant asymmetry measure [0, 1]

    def to_dict(self, round_decimals: int = 4) -> Dict[str, float]:
        return {
            "support": round(self.support, round_decimals),
            "antecedent_support": round(self.antecedent_support, round_decimals),
            "consequent_support": round(self.consequent_support, round_decimals),
            "confidence": round(self.confidence, round_decimals),
            "lift": round(self.lift, round_decimals),
            "conviction": round(self.conviction, round_decimals) if not math.isinf(self.conviction) else 9999.0,
            "leverage": round(self.leverage, round_decimals),
            "zhangs_metric": round(self.zhangs_metric, round_decimals),
            "kulczynski": round(self.kulczynski, round_decimals),
            "imbalance_ratio": round(self.imbalance_ratio, round_decimals),
        }


def compute_rule_metrics(
    supp_auc: float,
    supp_a: float,
    supp_c: float,
    epsilon: float = 1e-12
) -> RuleMetrics:
    """
    Computes 7 comprehensive interestingness metrics for rule A -> C:
    - Support: P(A U C)
    - Confidence: P(A U C) / P(A)
    - Lift: P(A U C) / (P(A) * P(C))
    - Conviction: (1 - P(C)) / (1 - Conf)
    - Leverage: P(A U C) - P(A) * P(C)
    - Zhang's Metric: (P(A U C) - P(A)*P(C)) / max(P(A U C)*(1 - P(A)), P(A)*(P(C) - P(A U C))) [if >= 0]
    - Kulczynski: 0.5 * (conf(A->C) + conf(C->A))
    - Imbalance Ratio: |P(A) - P(C)| / (P(A) + P(C) - P(A U C))
    """
    # 1. Support
    support = float(supp_auc)
    ante_supp = float(supp_a)
    cons_supp = float(supp_c)

    # 2. Confidence: P(A U C) / P(A)
    confidence = support / max(ante_supp, epsilon)
    confidence = min(1.0, max(0.0, confidence))

    # 3. Lift: Conf(A -> C) / P(C)
    lift = support / max(ante_supp * cons_supp, epsilon)

    # 4. Conviction: (1 - P(C)) / (1 - Conf)
    if abs(1.0 - confidence) < 1e-7:
        conviction = float("inf")
    else:
        conviction = (1.0 - cons_supp) / (1.0 - confidence)
        conviction = max(0.0, conviction)

    # 5. Leverage: P(A U C) - P(A) * P(C)
    leverage = support - (ante_supp * cons_supp)

    # 6. Zhang's Metric (Zhang, 2000)
    # Measures degree of association normalized between -1 and +1
    numerator = support - (ante_supp * cons_supp)
    if abs(numerator) < epsilon:
        zhang = 0.0
    elif numerator > 0:
        denom = max(support * (1.0 - ante_supp), ante_supp * (cons_supp - support))
        zhang = numerator / denom if denom > epsilon else 0.0
    else:
        denom = max(ante_supp * (support - cons_supp), support * (1.0 - ante_supp))
        zhang = numerator / denom if denom > epsilon else 0.0
    zhang = max(-1.0, min(1.0, zhang))

    # 7. Kulczynski (Kulczynski 1927; Wu et al. 2007)
    # Null-invariant metric: 0.5 * (P(A U C)/P(A) + P(A U C)/P(C))
    conf_c_to_a = support / max(cons_supp, epsilon)
    conf_c_to_a = min(1.0, max(0.0, conf_c_to_a))
    kulczynski = 0.5 * (confidence + conf_c_to_a)
    kulczynski = max(0.0, min(1.0, kulczynski))

    # 8. Imbalance Ratio (IR)
    # Measures transaction frequency imbalance between antecedent and consequent
    ir_denom = ante_supp + cons_supp - support
    if ir_denom > epsilon:
        imbalance_ratio = abs(ante_supp - cons_supp) / ir_denom
    else:
        imbalance_ratio = 0.0
    imbalance_ratio = max(0.0, min(1.0, imbalance_ratio))

    return RuleMetrics(
        support=support,
        antecedent_support=ante_supp,
        consequent_support=cons_supp,
        confidence=confidence,
        lift=lift,
        conviction=conviction,
        leverage=leverage,
        zhangs_metric=zhang,
        kulczynski=kulczynski,
        imbalance_ratio=imbalance_ratio
    )
