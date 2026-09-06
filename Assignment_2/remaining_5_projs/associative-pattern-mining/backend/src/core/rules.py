"""Association rule derivation, interestingness evaluation, and redundancy pruning."""

from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from typing import List, Dict, Set, Optional, Tuple
from .metrics import RuleMetrics, compute_rule_metrics
from ..algorithms.base import FrequentItemset


@dataclass(frozen=True)
class AssociationRule:
    """Represents a discovered association rule A -> C with comprehensive metrics."""
    antecedent: frozenset[str]
    consequent: frozenset[str]
    metrics: RuleMetrics

    @property
    def antecedent_list(self) -> List[str]:
        return sorted(list(self.antecedent))

    @property
    def consequent_list(self) -> List[str]:
        return sorted(list(self.consequent))

    @property
    def rule_string(self) -> str:
        ante_str = "{" + ", ".join(self.antecedent_list) + "}"
        cons_str = "{" + ", ".join(self.consequent_list) + "}"
        return f"{ante_str} => {cons_str}"

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "antecedent": self.antecedent_list,
            "consequent": self.consequent_list,
            "rule_string": self.rule_string,
            "length_antecedent": len(self.antecedent),
            "length_consequent": len(self.consequent),
            "length_total": len(self.antecedent) + len(self.consequent),
        }
        data.update(self.metrics.to_dict())
        return data


class RuleGenerator:
    """Derives valid association rules from frequent itemsets according to metrics and pruning constraints."""

    @staticmethod
    def generate_rules(
        frequent_itemsets: List[FrequentItemset],
        min_confidence: float = 0.5,
        min_lift: float = 1.0,
        max_consequent_len: int = 1,
        prune_redundant: bool = True
    ) -> List[AssociationRule]:
        """
        Generates association rules from frequent itemsets.
        Itemsets must be indexed by frozenset for rapid O(1) support lookups.
        """
        if not frequent_itemsets:
            return []

        # Map itemsets to support
        support_map: Dict[frozenset[str], float] = {
            itemset.items: itemset.support for itemset in frequent_itemsets
        }

        raw_rules: List[AssociationRule] = []

        # Iterate over itemsets of length >= 2
        for itemset in frequent_itemsets:
            items = itemset.items
            k = len(items)
            if k < 2:
                continue

            supp_x = itemset.support

            # Generate all non-empty proper subsets as antecedents
            # If max_consequent_len == 1, antecedent length is k - 1
            for ante_len in range(1, k):
                cons_len = k - ante_len
                if max_consequent_len is not None and cons_len > max_consequent_len:
                    continue

                for ante_tuple in combinations(items, ante_len):
                    ante = frozenset(ante_tuple)
                    cons = items.difference(ante)

                    supp_a = support_map.get(ante)
                    supp_c = support_map.get(cons)

                    # If support is not precomputed (e.g. if cons was infrequent on its own), skip
                    if supp_a is None or supp_c is None:
                        continue

                    # Confidence check early exit
                    conf = supp_x / supp_a
                    if conf < min_confidence:
                        continue

                    # Lift check early exit
                    lift = supp_x / (supp_a * supp_c)
                    if lift < min_lift:
                        continue

                    metrics = compute_rule_metrics(
                        supp_auc=supp_x,
                        supp_a=supp_a,
                        supp_c=supp_c
                    )

                    raw_rules.append(AssociationRule(
                        antecedent=ante,
                        consequent=cons,
                        metrics=metrics
                    ))

        if prune_redundant:
            return RuleGenerator.prune_redundant_rules(raw_rules)

        return raw_rules

    @staticmethod
    def prune_redundant_rules(rules: List[AssociationRule]) -> List[AssociationRule]:
        """
        Prunes redundant association rules.
        A rule A' -> C is redundant if there exists A -> C with A c A' such that Conf(A -> C) >= Conf(A' -> C).
        The more general antecedent A is strictly more parsimonious and equally or more reliable.
        """
        if not rules:
            return []

        # Group rules by consequent
        rules_by_cons: Dict[frozenset[str], List[AssociationRule]] = {}
        for r in rules:
            rules_by_cons.setdefault(r.consequent, []).append(r)

        non_redundant: List[AssociationRule] = []

        for cons, r_list in rules_by_cons.items():
            # Sort by antecedent length ascending, then confidence descending
            r_list.sort(key=lambda r: (len(r.antecedent), -r.metrics.confidence))
            kept: List[AssociationRule] = []

            for candidate in r_list:
                is_redundant = False
                for existing in kept:
                    # If existing has a subset antecedent and equal or higher confidence
                    if existing.antecedent.issubset(candidate.antecedent):
                        if existing.metrics.confidence >= candidate.metrics.confidence:
                            is_redundant = True
                            break
                if not is_redundant:
                    kept.append(candidate)

            non_redundant.extend(kept)

        # Sort final rules by lift descending, then confidence descending
        non_redundant.sort(key=lambda r: (-r.metrics.lift, -r.metrics.confidence))
        return non_redundant
