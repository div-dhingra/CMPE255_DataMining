"""Abstract base interface and dataclasses for association pattern mining algorithms."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


@dataclass(frozen=True)
class FrequentItemset:
    """Represents a discovered frequent itemset."""
    items: frozenset[str]
    support: float
    count: int
    length: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "items": sorted(list(self.items)),
            "support": round(self.support, 6),
            "count": self.count,
            "length": self.length
        }


@dataclass
class MiningResult:
    """Encapsulates the complete execution telemetry and discovered itemsets."""
    algorithm_name: str
    itemsets: List[FrequentItemset]
    execution_time_ms: float
    num_transactions: int
    min_support: float
    max_len: int
    memory_mb: Optional[float] = None
    parameters: Optional[Dict[str, Any]] = None

    @property
    def total_itemsets(self) -> int:
        return len(self.itemsets)

    def itemsets_by_length(self) -> Dict[int, int]:
        dist: Dict[int, int] = {}
        for itemset in self.itemsets:
            dist[itemset.length] = dist.get(itemset.length, 0) + 1
        return dict(sorted(dist.items()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm_name,
            "execution_time_ms": round(self.execution_time_ms, 2),
            "num_transactions": self.num_transactions,
            "min_support": self.min_support,
            "max_len": self.max_len,
            "total_itemsets": self.total_itemsets,
            "itemsets_by_length": self.itemsets_by_length(),
            "itemsets": [itemset.to_dict() for itemset in self.itemsets]
        }


class MiningAlgorithm(ABC):
    """Abstract base class for all frequent pattern mining algorithms."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def mine(
        self,
        transactions: List[frozenset[str]],
        min_support: float,
        max_len: int = 5
    ) -> MiningResult:
        """Executes frequent pattern mining on the provided transaction database."""
        pass
