"""Tests for association rule generation, pruning, and shopping cart recommendations."""

from backend.src.core.rules import RuleGenerator
from backend.src.core.recommendations import CartRecommender
from backend.src.algorithms.fpgrowth import FPGrowthMiner
from backend.src.core.dataset import TransactionDataset


def test_rule_generation_and_redundancy_pruning(sample_transactions):
    miner = FPGrowthMiner()
    mining_res = miner.mine(sample_transactions, min_support=0.25, max_len=3)

    raw_rules = RuleGenerator.generate_rules(
        frequent_itemsets=mining_res.itemsets,
        min_confidence=0.40,
        min_lift=1.0,
        prune_redundant=False
    )
    pruned_rules = RuleGenerator.generate_rules(
        frequent_itemsets=mining_res.itemsets,
        min_confidence=0.40,
        min_lift=1.0,
        prune_redundant=True
    )

    assert len(raw_rules) >= len(pruned_rules)
    for r in pruned_rules:
        assert r.metrics.confidence >= 0.40
        assert r.metrics.lift >= 1.0


def test_cart_recommendations_with_matching_rules(sample_transactions):
    ds = TransactionDataset(transactions=[set(t) for t in sample_transactions])
    miner = FPGrowthMiner()
    mining_res = miner.mine(sample_transactions, min_support=0.20, max_len=3)
    rules = RuleGenerator.generate_rules(mining_res.itemsets, min_confidence=0.30, min_lift=1.0)

    recommender = CartRecommender(rules=rules, dataset=ds)

    # Cart containing Bread & Butter
    rec_res = recommender.recommend(cart_items=["Bread", "Butter"], top_k=3)

    assert rec_res["num_cart_items"] == 2
    assert rec_res["recommendations_count"] > 0
    # Items in cart must NOT be recommended again
    rec_items = [r["item"] for r in rec_res["recommendations"]]
    assert "Bread" not in rec_items
    assert "Butter" not in rec_items

    # Verify explanations exist
    for r in rec_res["recommendations"]:
        assert len(r["explanation"]) > 10


def test_cart_recommendations_cold_start(sample_transactions):
    ds = TransactionDataset(transactions=[set(t) for t in sample_transactions])
    recommender = CartRecommender(rules=[], dataset=ds)

    # Empty cart
    rec_res = recommender.recommend(cart_items=[], top_k=4)
    assert rec_res["num_cart_items"] == 0
    assert rec_res["mode"] == "cold_start_popular"
    assert len(rec_res["recommendations"]) > 0
