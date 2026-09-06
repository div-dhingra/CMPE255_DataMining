"""Tests for dataset ingestion, cleaning, one-hot encoding, and synthetic generation."""

import pandas as pd
from backend.src.core.dataset import TransactionDataset, DatasetLoader, SyntheticTransactionGenerator


def test_transaction_dataset_properties(sample_transactions):
    ds = TransactionDataset(transactions=[set(t) for t in sample_transactions])
    assert ds.num_transactions == 8
    assert ds.num_unique_items == 6

    counts = ds.item_counts
    assert counts["Bread"] == 5
    assert counts["Butter"] == 5
    assert counts["Milk"] == 5

    stats = ds.get_summary_stats()
    assert stats["num_transactions"] == 8
    assert stats["num_unique_items"] == 6
    assert stats["min_basket_size"] == 2
    assert stats["max_basket_size"] == 4
    assert stats["density_percent"] > 0


def test_one_hot_dataframe_generation(sample_transactions):
    ds = TransactionDataset(transactions=[set(t) for t in sample_transactions])
    df = ds.to_one_hot_df()

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (8, 6)
    assert df.dtypes.iloc[0] == bool
    # First row has Milk, Bread, Butter
    assert df.iloc[0]["Milk"] == True
    assert df.iloc[0]["Bread"] == True
    assert df.iloc[0]["Butter"] == True
    assert df.iloc[0]["Beer"] == False


def test_cancellation_and_return_cleaning():
    # Construct raw dataframe with returns (Invoice starting with 'C' and negative quantity)
    raw_data = {
        "InvoiceNo": ["5001", "5001", "5002", "C5003", "5004"],
        "Description": ["Item A", "Item B", "Item A", "Item A", "POSTAGE"],
        "Quantity": [2, 1, 3, -1, 1],
        "UnitPrice": [10.0, 5.0, 10.0, 10.0, 2.0]
    }
    df = pd.DataFrame(raw_data)
    ds = DatasetLoader.parse_dataframe(df, source_name="test_raw")

    # Invoices remaining should only be 5001 and 5002 (C5003 cancelled, 5004 has POSTAGE which is scrubbed)
    assert ds.num_transactions == 2
    assert "POSTAGE" not in ds.unique_items
    assert ds.unique_items == ["Item A", "Item B"]


def test_market_basket_tid_schema_loading():
    raw_data = {
        "TransactionID": ["T1", "T1", "T2", "T3", "T3"],
        "Item": ["Apple", "Banana", "Apple", "Banana", "Orange"]
    }
    df = pd.DataFrame(raw_data)
    ds = DatasetLoader.parse_dataframe(df, source_name="test_tid")

    assert ds.num_transactions == 3
    assert ds.num_unique_items == 3
    assert set(ds.unique_items) == {"Apple", "Banana", "Orange"}


def test_synthetic_transaction_generator():
    synth = SyntheticTransactionGenerator.generate(num_transactions=150, avg_basket_size=3.5, seed=42)

    assert synth.num_transactions == 150
    assert synth.num_unique_items >= 5
    stats = synth.get_summary_stats()
    assert 2.0 <= stats["avg_basket_size"] <= 6.0
    assert stats["metadata"]["source"] == "synthetic_generator"
