"""Market basket recommendation engine with association rule explanations and uplift scoring."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Set, Any, Optional
from .rules import AssociationRule
from .dataset import TransactionDataset


@dataclass
class CartRecommendation:
    """Individual item recommendation for a user cart."""
    item: str
    score: float
    trigger_rule: str
    trigger_antecedent: List[str]
    confidence: float
    lift: float
    kulczynski: float
    support: float
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item": self.item,
            "score": round(self.score, 4),
            "trigger_rule": self.trigger_rule,
            "trigger_antecedent": self.trigger_antecedent,
            "confidence": round(self.confidence, 4),
            "lift": round(self.lift, 4),
            "kulczynski": round(self.kulczynski, 4),
            "support": round(self.support, 4),
            "explanation": self.explanation
        }


class CartRecommender:
    """Generates recommendations for an active shopping cart from mined rules."""

    def __init__(self, rules: List[AssociationRule], dataset: Optional[TransactionDataset] = None):
        self.rules = rules
        self.dataset = dataset

    def recommend(
        self,
        cart_items: List[str],
        top_k: int = 6,
        min_lift: float = 1.05
    ) -> Dict[str, Any]:
        cart_set = set(cart_items)

        if not cart_set:
            return self._cold_start_recommendations(top_k=top_k)

        # Find rules where antecedent is a subset of the cart
        matching_rules = [
            r for r in self.rules
            if r.antecedent.issubset(cart_set) and r.metrics.lift >= min_lift
        ]

        # Candidate item -> Best matching rule info
        candidates: Dict[str, Dict[str, Any]] = {}

        for rule in matching_rules:
            for item in rule.consequent:
                if item in cart_set:
                    continue  # Already in cart

                # Score combines lift and confidence
                score = rule.metrics.confidence * (1.0 + (min(rule.metrics.lift, 10.0) / 10.0))

                if item not in candidates or score > candidates[item]["score"]:
                    candidates[item] = {
                        "score": score,
                        "rule": rule,
                        "confidence": rule.metrics.confidence,
                        "lift": rule.metrics.lift,
                        "kulczynski": rule.metrics.kulczynski,
                        "support": rule.metrics.support,
                        "antecedent": rule.antecedent_list,
                        "rule_str": rule.rule_string
                    }

        sorted_candidates = sorted(candidates.items(), key=lambda x: x[1]["score"], reverse=True)[:top_k]

        results: List[CartRecommendation] = []
        for item, meta in sorted_candidates:
            ante_joined = ", ".join(meta["antecedent"])
            explanation = (
                f"Recommended based on '{ante_joined}' in your basket. "
                f"Shoppers who bought these also purchased '{item}' "
                f"({round(meta['confidence'] * 100, 1)}% of the time, {round(meta['lift'], 2)}x higher than average)."
            )
            results.append(CartRecommendation(
                item=item,
                score=meta["score"],
                trigger_rule=meta["rule_str"],
                trigger_antecedent=meta["antecedent"],
                confidence=meta["confidence"],
                lift=meta["lift"],
                kulczynski=meta["kulczynski"],
                support=meta["support"],
                explanation=explanation
            ))

        # If not enough rule-based recommendations, fill with cold-start popular items
        if len(results) < top_k and self.dataset:
            popular = self._get_popular_items(exclude=cart_set.union({r.item for r in results}), limit=top_k - len(results))
            results.extend(popular)

        # Compute projected cart basket lift
        avg_lift = sum(r.lift for r in results) / len(results) if results else 1.0
        projected_cross_sell_pct = min(100.0, max(0.0, (avg_lift - 1.0) * 25.0))

        return {
            "cart_items": cart_items,
            "num_cart_items": len(cart_items),
            "recommendations_count": len(results),
            "recommendations": [r.to_dict() for r in results],
            "projected_cross_sell_uplift_pct": round(projected_cross_sell_pct, 1),
            "matched_rules_count": len(matching_rules)
        }

    def _cold_start_recommendations(self, top_k: int = 6) -> Dict[str, Any]:
        """Provides baseline popular recommendations when cart is empty."""
        results: List[CartRecommendation] = []
        if self.dataset:
            top_items = list(self.dataset.item_counts.items())[:top_k]
            n_tx = max(1, self.dataset.num_transactions)
            for item, count in top_items:
                supp = count / n_tx
                results.append(CartRecommendation(
                    item=item,
                    score=supp,
                    trigger_rule="[Global Popularity Baseline]",
                    trigger_antecedent=[],
                    confidence=supp,
                    lift=1.0,
                    kulczynski=supp,
                    support=supp,
                    explanation=f"Trending catalog item ({round(supp * 100, 1)}% market penetration)."
                ))

        return {
            "cart_items": [],
            "num_cart_items": 0,
            "recommendations_count": len(results),
            "recommendations": [r.to_dict() for r in results],
            "projected_cross_sell_uplift_pct": 0.0,
            "matched_rules_count": 0,
            "mode": "cold_start_popular"
        }

    def _get_popular_items(self, exclude: Set[str], limit: int) -> List[CartRecommendation]:
        recs: List[CartRecommendation] = []
        if not self.dataset:
            return recs
        n_tx = max(1, self.dataset.num_transactions)
        for item, count in self.dataset.item_counts.items():
            if item in exclude:
                continue
            supp = count / n_tx
            recs.append(CartRecommendation(
                item=item,
                score=supp * 0.5,
                trigger_rule="[Catalog Popularity Fallback]",
                trigger_antecedent=[],
                confidence=supp,
                lift=1.0,
                kulczynski=supp,
                support=supp,
                explanation=f"Top-selling catalog product ({round(supp * 100, 1)}% baseline frequency)."
            ))
            if len(recs) >= limit:
                break
        return recs
