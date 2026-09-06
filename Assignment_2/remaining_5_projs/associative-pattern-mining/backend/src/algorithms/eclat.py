"""ECLAT algorithm for frequent pattern mining (Zaki, 2000)."""

from __future__ import annotations
import time
from typing import List, Dict, Set, Tuple
from .base import MiningAlgorithm, MiningResult, FrequentItemset


class EclatMiner(MiningAlgorithm):
    """
    Equivalence Class Clustering and bottom-up Lattice Traversal (ECLAT).
    Uses vertical data representation (TID sets) and set intersection.
    Reference: Zaki (TKDE 2000) - "Scalable Algorithms for Association Mining".
    """

    def __init__(self):
        super().__init__(name="ECLAT")

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

        # 1. Build vertical database: Item -> Set of Transaction IDs
        vertical_db: Dict[str, Set[int]] = {}
        for tid, t in enumerate(transactions):
            for item in t:
                if item not in vertical_db:
                    vertical_db[item] = set()
                vertical_db[item].add(tid)

        # 2. Filter frequent 1-itemsets
        frequent_singletons: List[Tuple[str, Set[int]]] = []
        discovered_patterns: Dict[frozenset[str], int] = {}

        for item, tids in vertical_db.items():
            cnt = len(tids)
            if cnt >= min_count:
                pattern = frozenset([item])
                discovered_patterns[pattern] = cnt
                frequent_singletons.append((item, tids))

        # Sort singletons by frequency descending
        frequent_singletons.sort(key=lambda x: len(x[1]), reverse=True)

        # 3. Depth-first lattice traversal
        self._recurse_eclat(
            prefix_items=[],
            candidates=frequent_singletons,
            min_count=min_count,
            max_len=max_len,
            discovered_patterns=discovered_patterns
        )

        discovered_itemsets: List[FrequentItemset] = [
            FrequentItemset(
                items=pattern,
                support=cnt / n_tx,
                count=cnt,
                length=len(pattern)
            )
            for pattern, cnt in discovered_patterns.items()
            if 1 <= len(pattern) <= max_len
        ]

        discovered_itemsets.sort(key=lambda x: (x.length, -x.support))

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

    def _recurse_eclat(
        self,
        prefix_items: List[str],
        candidates: List[Tuple[str, Set[int]]],
        min_count: int,
        max_len: int,
        discovered_patterns: Dict[frozenset[str], int]
    ) -> None:
        if len(prefix_items) + 1 >= max_len:
            # We can still form pairs from candidates, but don't recurse deeper
            for i in range(len(candidates)):
                item_i, tids_i = candidates[i]
                for j in range(i + 1, len(candidates)):
                    item_j, tids_j = candidates[j]
                    common_tids = tids_i.intersection(tids_j)
                    cnt = len(common_tids)
                    if cnt >= min_count:
                        new_pattern = frozenset(prefix_items + [item_i, item_j])
                        discovered_patterns[new_pattern] = cnt
            return

        for i in range(len(candidates)):
            item_i, tids_i = candidates[i]
            next_candidates: List[Tuple[str, Set[int]]] = []

            for j in range(i + 1, len(candidates)):
                item_j, tids_j = candidates[j]
                common_tids = tids_i.intersection(tids_j)
                cnt = len(common_tids)
                if cnt >= min_count:
                    new_pattern = frozenset(prefix_items + [item_i, item_j])
                    discovered_patterns[new_pattern] = cnt
                    next_candidates.append((item_j, common_tids))

            if next_candidates and len(prefix_items) + 2 < max_len:
                self._recurse_eclat(
                    prefix_items=prefix_items + [item_i],
                    candidates=next_candidates,
                    min_count=min_count,
                    max_len=max_len,
                    discovered_patterns=discovered_patterns
                )
