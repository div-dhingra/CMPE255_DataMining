"""Tests verifying Apriori, FP-Growth, and ECLAT algorithms and their consistency."""

import pytest
from backend.src.algorithms.apriori import AprioriMiner
from backend.src.algorithms.fpgrowth import FPGrowthMiner
from backend.src.algorithms.eclat import EclatMiner


@pytest.mark.parametrize("min_supp", [0.25, 0.35, 0.50])
def test_all_algorithms_produce_identical_itemsets(sample_transactions, min_supp):
    ap = AprioriMiner().mine(sample_transactions, min_support=min_supp, max_len=4)
    fp = FPGrowthMiner().mine(sample_transactions, min_support=min_supp, max_len=4)
    ec = EclatMiner().mine(sample_transactions, min_support=min_supp, max_len=4)

    ap_map = {x.items: x.count for x in ap.itemsets}
    fp_map = {x.items: x.count for x in fp.itemsets}
    ec_map = {x.items: x.count for x in ec.itemsets}

    assert ap_map == fp_map, f"Apriori and FP-Growth discrepancy at supp={min_supp}"
    assert ap_map == ec_map, f"Apriori and ECLAT discrepancy at supp={min_supp}"
    assert len(ap_map) > 0


def test_max_length_constraint(sample_transactions):
    for max_l in [1, 2, 3]:
        for miner in [AprioriMiner(), FPGrowthMiner(), EclatMiner()]:
            res = miner.mine(sample_transactions, min_support=0.25, max_len=max_l)
            for itemset in res.itemsets:
                assert itemset.length <= max_l, f"{miner.name} exceeded max_len={max_l}"


def test_empty_transactions():
    for miner in [AprioriMiner(), FPGrowthMiner(), EclatMiner()]:
        res = miner.mine([], min_support=0.1, max_len=3)
        assert res.total_itemsets == 0
        assert res.num_transactions == 0


def test_high_support_threshold(sample_transactions):
    for miner in [AprioriMiner(), FPGrowthMiner(), EclatMiner()]:
        res = miner.mine(sample_transactions, min_support=0.99, max_len=3)
        assert res.total_itemsets == 0
