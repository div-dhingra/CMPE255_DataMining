"""FP-Growth (Frequent Pattern Tree) algorithm (Han, Pei, & Yin, 2000)."""

from __future__ import annotations
import time
from typing import List, Dict, Optional, Set
from .base import MiningAlgorithm, MiningResult, FrequentItemset


class FPNode:
    """Node in the Frequent Pattern Tree."""
    __slots__ = ("item", "count", "parent", "children", "node_link")

    def __init__(self, item: Optional[str], count: int = 1, parent: Optional[FPNode] = None):
        self.item: Optional[str] = item
        self.count: int = count
        self.parent: Optional[FPNode] = parent
        self.children: Dict[str, FPNode] = {}
        self.node_link: Optional[FPNode] = None  # Pointer to next node with same item in header table


class FPGrowthMiner(MiningAlgorithm):
    """
    Frequent Pattern Tree without candidate generation.
    Reference: Han, Pei, & Yin (SIGMOD 2000) - "Mining Frequent Patterns without Candidate Generation".
    """

    def __init__(self):
        super().__init__(name="FP-Growth")

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

        # 1. First scan: count item frequencies and build F-List
        item_counts: Dict[str, int] = {}
        for t in transactions:
            for item in t:
                item_counts[item] = item_counts.get(item, 0) + 1

        # Filter items meeting min_support
        frequent_items = {item: cnt for item, cnt in item_counts.items() if cnt >= min_count}
        if not frequent_items:
            return MiningResult(
                algorithm_name=self.name,
                itemsets=[],
                execution_time_ms=(time.perf_counter() - start_time) * 1000.0,
                num_transactions=n_tx,
                min_support=min_support,
                max_len=max_len
            )

        # Ordering rank for F-List (descending frequency, then lexicographical tie-breaker)
        item_rank = {
            item: (-frequent_items[item], item)
            for item in frequent_items
        }

        # 2. Second scan: construct FP-Tree
        root = FPNode(item=None, count=0, parent=None)
        header_table: Dict[str, List[Optional[FPNode]]] = {
            item: [frequent_items[item], None]  # [count, head_pointer]
            for item in frequent_items
        }

        for t in transactions:
            # Filter and sort transaction items according to F-List
            sorted_items = [item for item in t if item in frequent_items]
            sorted_items.sort(key=lambda x: item_rank[x])

            if sorted_items:
                self._insert_tree(sorted_items, root, header_table)

        # 3. Recursively mine FP-Tree
        discovered_patterns: Dict[frozenset[str], int] = {}
        self._mine_tree(header_table, min_count, frozenset(), discovered_patterns, max_len)

        # Convert to FrequentItemset instances
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

        # Sort by length ascending, support descending
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

    def _insert_tree(
        self,
        items: List[str],
        current_node: FPNode,
        header_table: Dict[str, List[Optional[FPNode]]]
    ) -> None:
        first_item = items[0]
        if first_item in current_node.children:
            child = current_node.children[first_item]
            child.count += 1
        else:
            child = FPNode(item=first_item, count=1, parent=current_node)
            current_node.children[first_item] = child

            # Update header table linked list
            if header_table[first_item][1] is None:
                header_table[first_item][1] = child
            else:
                current_link = header_table[first_item][1]
                while current_link.node_link is not None:
                    current_link = current_link.node_link
                current_link.node_link = child

        # Recursive insert remaining items down the tree
        if len(items) > 1:
            self._insert_tree(items[1:], child, header_table)

    def _mine_tree(
        self,
        header_table: Dict[str, List[Optional[FPNode]]],
        min_count: int,
        prefix: frozenset[str],
        patterns: Dict[frozenset[str], int],
        max_len: int
    ) -> None:
        # Sort header table items ascending by frequency (bottom-up mining)
        sorted_items = sorted(header_table.keys(), key=lambda x: header_table[x][0])

        for item in sorted_items:
            item_count = header_table[item][0]
            new_pattern = prefix.union([item])

            if len(new_pattern) > max_len:
                continue

            patterns[new_pattern] = item_count

            if len(new_pattern) >= max_len:
                continue

            # Build conditional pattern base for `item`
            cond_pattern_base: List[List[str]] = []
            cond_pattern_counts: List[int] = []

            node = header_table[item][1]
            while node is not None:
                path: List[str] = []
                parent = node.parent
                while parent is not None and parent.item is not None:
                    path.append(parent.item)
                    parent = parent.parent

                if path:
                    cond_pattern_base.append(path)
                    cond_pattern_counts.append(node.count)

                node = node.node_link

            # Construct conditional FP-Tree
            cond_item_counts: Dict[str, int] = {}
            for path, count in zip(cond_pattern_base, cond_pattern_counts):
                for p_item in path:
                    cond_item_counts[p_item] = cond_item_counts.get(p_item, 0) + count

            # Filter frequent items in conditional tree
            cond_frequent = {p: cnt for p, cnt in cond_item_counts.items() if cnt >= min_count}
            if not cond_frequent:
                continue

            cond_rank = {p: (-cond_frequent[p], p) for p in cond_frequent}

            # Build conditional tree
            cond_root = FPNode(item=None, count=0, parent=None)
            cond_header: Dict[str, List[Optional[FPNode]]] = {
                p: [cond_frequent[p], None] for p in cond_frequent
            }

            for path, count in zip(cond_pattern_base, cond_pattern_counts):
                filtered_path = [p for p in path if p in cond_frequent]
                filtered_path.sort(key=lambda x: cond_rank[x])
                if filtered_path:
                    self._insert_cond_tree(filtered_path, count, cond_root, cond_header)

            # Recurse
            self._mine_tree(cond_header, min_count, new_pattern, patterns, max_len)

    def _insert_cond_tree(
        self,
        items: List[str],
        count: int,
        current_node: FPNode,
        header_table: Dict[str, List[Optional[FPNode]]]
    ) -> None:
        first_item = items[0]
        if first_item in current_node.children:
            child = current_node.children[first_item]
            child.count += count
        else:
            child = FPNode(item=first_item, count=count, parent=current_node)
            current_node.children[first_item] = child

            if header_table[first_item][1] is None:
                header_table[first_item][1] = child
            else:
                curr = header_table[first_item][1]
                while curr.node_link is not None:
                    curr = curr.node_link
                curr.node_link = child

        if len(items) > 1:
            self._insert_cond_tree(items[1:], count, child, header_table)
