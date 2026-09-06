"""Apriori algorithm for frequent pattern mining (Agrawal & Srikant, 1994)."""

from __future__ import annotations
import time
from itertools import combinations
from typing import List, Dict, Set, FrozenSet
from .base import MiningAlgorithm, MiningResult, FrequentItemset


class AprioriMiner(MiningAlgorithm):
    """
    Classic Apriori implementation with downward-closure candidate pruning.
    Reference: Agrawal & Srikant (VLDB 1994) - "Fast Algorithms for Mining Association Rules".
    """

    def __init__(self):
        super().__init__(name="Apriori")

    def mine(
        self,
        transactions: List[frozenset[str]],
        min_support: float,
        max_len: int = 5
    ) -> MiningResult:
        start_time = time.perf_counter()
        n_tx = len(transactions)

        if n_tx == 0 or min_support <= 0.0:
            return MiningResult(
                algorithm_name=self.name,
                itemsets=[],
                execution_time_ms=0.0,
                num_transactions=n_tx,
                min_support=min_support,
                max_len=max_len
            )

        min_count = int(min_support * n_tx)
        if min_count < 1:
            min_count = 1

        discovered_itemsets: List[FrequentItemset] = []

        # 1. Candidate 1-itemsets (L1)
        item_counts: Dict[str, int] = {}
        for t in transactions:
            for item in t:
                item_counts[item] = item_counts.get(item, 0) + 1

        # Filter L1
        current_frequent: Dict[frozenset[str], int] = {}
        for item, count in item_counts.items():
            if count >= min_count:
                f_itemset = frozenset([item])
                current_frequent[f_itemset] = count
                discovered_itemsets.append(FrequentItemset(
                    items=f_itemset,
                    support=count / n_tx,
                    count=count,
                    length=1
                ))

        k = 2
        while current_frequent and k <= max_len:
            # 2. Generate Candidate k-itemsets: Join step L_{k-1} * L_{k-1}
            candidates = self._generate_candidates(list(current_frequent.keys()), k)

            if not candidates:
                break

            # 3. Prune Candidates whose (k-1) subsets are not in current_frequent
            pruned_candidates = self._prune_candidates(candidates, set(current_frequent.keys()), k)

            if not pruned_candidates:
                break

            # 4. Count support for pruned candidates
            candidate_counts: Dict[frozenset[str], int] = {c: 0 for c in pruned_candidates}
            for t in transactions:
                # Only check candidates that could be subsets of transaction t
                if len(t) < k:
                    continue
                for c in pruned_candidates:
                    if c.issubset(t):
                        candidate_counts[c] += 1

            # 5. Filter L_k
            next_frequent: Dict[frozenset[str], int] = {}
            for c, count in candidate_counts.items():
                if count >= min_count:
                    next_frequent[c] = count
                    discovered_itemsets.append(FrequentItemset(
                        items=c,
                        support=count / n_tx,
                        count=count,
                        length=k
                    ))

            current_frequent = next_frequent
            k += 1

        execution_time_ms = (time.perf_counter() - start_time) * 1000.0

        return MiningResult(
            algorithm_name=self.name,
            itemsets=discovered_itemsets,
            execution_time_ms=execution_time_ms,
            num_transactions=n_tx,
            min_support=min_support,
            max_len=max_len,
            parameters={"min_count": min_count}
        )

    def _generate_candidates(self, prev_frequent: List[frozenset[str]], k: int) -> Set[frozenset[str]]:
        """Generates candidates of length k from frequent itemsets of length k-1."""
        candidates: Set[frozenset[str]] = set()
        n = len(prev_frequent)

        # Sort itemsets into canonical ordered lists for efficient prefix joining
        ordered_prev = [sorted(list(x)) for x in prev_frequent]
        ordered_prev.sort()

        for i in range(n):
            for j in range(i + 1, n):
                # If they share the first k-2 items, union them
                if ordered_prev[i][:k - 2] == ordered_prev[j][:k - 2]:
                    candidate = frozenset(ordered_prev[i] + [ordered_prev[j][k - 2]])
                    candidates.add(candidate)
                else:
                    # Because they are sorted, if prefix differs, break inner loop early
                    break

        return candidates

    def _prune_candidates(
        self,
        candidates: Set[frozenset[str]],
        prev_frequent_set: Set[frozenset[str]],
        k: int
    ) -> Set[frozenset[str]]:
        """Prunes candidates where any (k-1) subset is not frequent (Apriori property)."""
        valid_candidates: Set[frozenset[str]] = set()
        for c in candidates:
            # Check all subsets of size k-1
            is_valid = True
            for subset in combinations(c, k - 1):
                if frozenset(subset) not in prev_frequent_set:
                    is_valid = False
                    break
            if is_valid:
                valid_candidates.add(c)
        return valid_candidates
