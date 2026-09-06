"""Central state container for dataset, mining models, and autoresearch engines."""

from __future__ import annotations
from pathlib import Path
from typing import Optional, List, Dict, Any
from .dataset import TransactionDataset, DatasetLoader, SyntheticTransactionGenerator
from .crisp_dm import CrispDMLifecycle
from .rules import AssociationRule, RuleGenerator
from .recommendations import CartRecommender
from ..algorithms.base import MiningResult
from ..algorithms.fpgrowth import FPGrowthMiner
from ..autoresearch.optimizer import HillClimbingOptimizer


class AppState:
    """Manages loaded dataset, active models, cached rules, and autoresearch session."""

    def __init__(self):
        self.lifecycle = CrispDMLifecycle()
        self.dataset: TransactionDataset = self._load_initial_dataset()
        self.cached_mining_result: Optional[MiningResult] = None
        self.cached_rules: List[AssociationRule] = []
        self.recommender: Optional[CartRecommender] = None
        self.optimizer: Optional[HillClimbingOptimizer] = None

        # Pre-mine baseline rules so dashboard is instantly interactive
        self.run_baseline_mining()

    def _load_initial_dataset(self) -> TransactionDataset:
        # Prefer Online Retail sample if available, fallback to Synthetic sample or generator
        retail_path = Path("data/online_retail_sample.csv")
        synth_path = Path("data/synthetic_sample.csv")

        if retail_path.exists():
            return DatasetLoader.load_from_csv(retail_path)
        elif synth_path.exists():
            return DatasetLoader.load_from_csv(synth_path)
        else:
            return SyntheticTransactionGenerator.generate(num_transactions=800)

    def run_baseline_mining(self) -> None:
        """Runs initial FP-Growth mining and rule generation."""
        miner = FPGrowthMiner()
        self.cached_mining_result = miner.mine(
            transactions=self.dataset.transactions,
            min_support=0.03,
            max_len=3
        )
        self.cached_rules = RuleGenerator.generate_rules(
            frequent_itemsets=self.cached_mining_result.itemsets,
            min_confidence=0.40,
            min_lift=1.10,
            max_consequent_len=1,
            prune_redundant=True
        )
        self.recommender = CartRecommender(rules=self.cached_rules, dataset=self.dataset)
        self.optimizer = HillClimbingOptimizer(dataset=self.dataset)

    def set_dataset(self, dataset: TransactionDataset) -> None:
        self.dataset = dataset
        self.run_baseline_mining()


# Global singleton instance
app_state = AppState()
