"""Transaction dataset ingestion, cleaning, one-hot encoding, and synthetic generation."""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Set
import random
import numpy as np
import pandas as pd


class TransactionDataset:
    """Encapsulates transaction data, statistics, and matrix representations."""

    def __init__(self, transactions: List[Set[str]], raw_df: Optional[pd.DataFrame] = None, metadata: Optional[Dict[str, Any]] = None):
        # Store transactions as list of frozen sets for fast immutable membership testing
        self.transactions: List[frozenset[str]] = [frozenset(t) for t in transactions if len(t) > 0]
        self.raw_df = raw_df
        self.metadata = metadata or {}
        self._item_counts: Optional[Dict[str, int]] = None
        self._one_hot_df: Optional[pd.DataFrame] = None

    @property
    def num_transactions(self) -> int:
        return len(self.transactions)

    @property
    def item_counts(self) -> Dict[str, int]:
        if self._item_counts is None:
            counts: Dict[str, int] = {}
            for t in self.transactions:
                for item in t:
                    counts[item] = counts.get(item, 0) + 1
            self._item_counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))
        return self._item_counts

    @property
    def unique_items(self) -> List[str]:
        return list(self.item_counts.keys())

    @property
    def num_unique_items(self) -> int:
        return len(self.unique_items)

    def get_summary_stats(self) -> Dict[str, Any]:
        """Calculates transaction and basket size statistical summaries."""
        if not self.transactions:
            return {
                "num_transactions": 0,
                "num_unique_items": 0,
                "total_items_purchased": 0,
                "avg_basket_size": 0.0,
                "median_basket_size": 0.0,
                "min_basket_size": 0,
                "max_basket_size": 0,
                "density_percent": 0.0,
                "top_items": []
            }

        sizes = [len(t) for t in self.transactions]
        n_tx = len(self.transactions)
        n_items = self.num_unique_items
        total_items = sum(sizes)
        density = (total_items / (n_tx * n_items) * 100.0) if (n_tx > 0 and n_items > 0) else 0.0

        top_items_list = [
            {
                "item": item,
                "count": count,
                "support": round(count / n_tx, 4),
                "frequency_percent": round((count / n_tx) * 100.0, 2)
            }
            for item, count in list(self.item_counts.items())[:20]
        ]

        return {
            "num_transactions": n_tx,
            "num_unique_items": n_items,
            "total_items_purchased": total_items,
            "avg_basket_size": round(float(np.mean(sizes)), 2),
            "median_basket_size": float(np.median(sizes)),
            "std_basket_size": round(float(np.std(sizes)), 2),
            "min_basket_size": int(np.min(sizes)),
            "max_basket_size": int(np.max(sizes)),
            "density_percent": round(density, 4),
            "sparsity_percent": round(100.0 - density, 4),
            "top_items": top_items_list,
            "metadata": self.metadata
        }

    def to_one_hot_df(self) -> pd.DataFrame:
        """Converts transactions into a one-hot boolean DataFrame."""
        if self._one_hot_df is None:
            all_items = self.unique_items
            item_to_idx = {item: i for i, item in enumerate(all_items)}
            matrix = np.zeros((len(self.transactions), len(all_items)), dtype=bool)

            for row_idx, t in enumerate(self.transactions):
                for item in t:
                    col_idx = item_to_idx.get(item)
                    if col_idx is not None:
                        matrix[row_idx, col_idx] = True

            self._one_hot_df = pd.DataFrame(matrix, columns=all_items)
        return self._one_hot_df


class DatasetLoader:
    """Robust parser and loader for retail and market basket CSVs."""

    @staticmethod
    def load_from_csv(file_path: str | Path, max_rows: Optional[int] = None) -> TransactionDataset:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found at: {path}")

        # Attempt to read CSV with pandas
        df = pd.read_csv(path, nrows=max_rows)
        return DatasetLoader.parse_dataframe(df, source_name=path.name)

    @staticmethod
    def parse_dataframe(df: pd.DataFrame, source_name: str = "custom") -> TransactionDataset:
        cols = [c.strip() for c in df.columns]
        df.columns = cols
        cols_lower = [c.lower() for c in cols]

        # Case 1: Online Retail Schema (InvoiceNo / StockCode / Description / Quantity)
        if "invoiceno" in cols_lower or "invoice" in cols_lower:
            inv_col = [c for c in cols if c.lower() in ("invoiceno", "invoice")][0]
            desc_col = [c for c in cols if c.lower() in ("description", "stockcode", "item", "product")][0]

            # Cleaning step: Filter cancelled invoices (start with 'C') and invalid descriptions
            clean_df = df.copy()
            clean_df[inv_col] = clean_df[inv_col].astype(str).str.strip()
            clean_df = clean_df[~clean_df[inv_col].str.startswith("C", na=False)]

            # Filter negative or zero quantities if Quantity column exists
            qty_col_matches = [c for c in cols if c.lower() in ("quantity", "qty")]
            if qty_col_matches:
                clean_df = clean_df[pd.to_numeric(clean_df[qty_col_matches[0]], errors="coerce") > 0]

            # Clean item names
            clean_df[desc_col] = clean_df[desc_col].astype(str).str.strip()
            clean_df = clean_df[
                clean_df[desc_col].str.len() > 1
            ]
            clean_df = clean_df[
                ~clean_df[desc_col].isin(["nan", "None", "POSTAGE", "Manual", "DOTCOM POSTAGE", "Adjust bad debt"])
            ]

            # Group by invoice
            grouped = clean_df.groupby(inv_col)[desc_col].apply(lambda s: set(s.dropna().unique()))
            transactions = [s for s in grouped if len(s) > 0]

            metadata = {
                "source": source_name,
                "type": "online_retail_tabular",
                "raw_rows": len(df),
                "clean_rows": len(clean_df),
                "invoices_parsed": len(transactions)
            }
            return TransactionDataset(transactions=transactions, raw_df=df, metadata=metadata)

        # Case 2: Market Basket Schema (TransactionID / Item)
        elif "transactionid" in cols_lower or "tid" in cols_lower or "basket_id" in cols_lower:
            tid_col = [c for c in cols if c.lower() in ("transactionid", "tid", "basket_id")][0]
            item_col = [c for c in cols if c.lower() in ("item", "product", "items", "description")][0]

            clean_df = df.dropna(subset=[tid_col, item_col]).copy()
            clean_df[item_col] = clean_df[item_col].astype(str).str.strip()
            grouped = clean_df.groupby(tid_col)[item_col].apply(lambda s: set(s.unique()))
            transactions = [s for s in grouped if len(s) > 0]

            metadata = {
                "source": source_name,
                "type": "market_basket_transactions",
                "raw_rows": len(df),
                "clean_rows": len(clean_df),
                "transactions_parsed": len(transactions)
            }
            return TransactionDataset(transactions=transactions, raw_df=df, metadata=metadata)

        # Case 3: Comma-separated items per row
        elif len(cols) == 1 or "items" in cols_lower:
            col = cols[0] if len(cols) == 1 else [c for c in cols if c.lower() == "items"][0]
            transactions = []
            for row in df[col].dropna():
                items = {item.strip() for item in str(row).split(",") if item.strip()}
                if items:
                    transactions.append(items)

            metadata = {
                "source": source_name,
                "type": "single_line_basket",
                "raw_rows": len(df),
                "transactions_parsed": len(transactions)
            }
            return TransactionDataset(transactions=transactions, raw_df=df, metadata=metadata)

        else:
            # Fallback: Treat each row as items from non-null string values
            transactions = []
            for _, row in df.iterrows():
                items = {str(val).strip() for val in row.dropna().values if str(val).strip() and str(val).strip().lower() not in ("nan", "none")}
                if items:
                    transactions.append(items)
            metadata = {
                "source": source_name,
                "type": "generic_matrix",
                "raw_rows": len(df),
                "transactions_parsed": len(transactions)
            }
            return TransactionDataset(transactions=transactions, raw_df=df, metadata=metadata)


class SyntheticTransactionGenerator:
    """Generates synthetic market basket transactions with planted affinity clusters and Zipf noise."""

    DEFAULT_VOCAB = [
        "Organic Milk", "Artisan Sourdough Bread", "Farmhouse Salted Butter", "Free Range Eggs",
        "Roasted Coffee Beans", "Pure Cane Sugar", "Earl Grey Tea", "Dark Chocolate Cookies",
        "MacBook Pro Laptop", "Wireless Magic Mouse", "Silicone Mousepad", "Noise-Cancelling Headphones", "Mechanical Keyboard",
        "Spaghetti Pasta", "Extra Virgin Olive Oil", "Organic Garlic Cloves", "San Marzano Tomato Sauce", "Parmigiano Reggiano",
        "Baby Diapers Size 3", "Unscented Baby Wipes", "Craft IPA Beer 6-Pack", "Organic Baby Formula",
        "Honeycrisp Apples", "Organic Bananas", "Navel Oranges", "Greek Yogurt", "Whole Grain Cereal",
        "Sparkling Water", "Tortilla Chips", "Organic Chunky Salsa", "Paper Napkins", "Bounty Paper Towels"
    ]

    PLANTED_AFFINITIES = [
        (["Artisan Sourdough Bread", "Farmhouse Salted Butter"], "Organic Milk", 0.88),
        (["Roasted Coffee Beans"], "Pure Cane Sugar", 0.82),
        (["MacBook Pro Laptop", "Wireless Magic Mouse"], "Silicone Mousepad", 0.92),
        (["Spaghetti Pasta", "Extra Virgin Olive Oil"], "Parmigiano Reggiano", 0.85),
        (["Baby Diapers Size 3"], "Craft IPA Beer 6-Pack", 0.74),
        (["Tortilla Chips"], "Organic Chunky Salsa", 0.89),
        (["Organic Bananas", "Greek Yogurt"], "Honeycrisp Apples", 0.78),
    ]

    @classmethod
    def generate(
        cls,
        num_transactions: int = 1000,
        avg_basket_size: float = 4.0,
        noise_level: float = 0.25,
        seed: Optional[int] = 42
    ) -> TransactionDataset:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        transactions: List[Set[str]] = []

        for _ in range(num_transactions):
            basket: Set[str] = set()

            # 1. Probabilistically plant affinities
            for ante, cons, prob in cls.PLANTED_AFFINITIES:
                if random.random() < 0.22:
                    basket.update(ante)
                    if random.random() < prob:
                        basket.add(cons)

            # 2. Add Poisson background items with Zipfian popularity
            target_size = max(1, int(np.random.poisson(avg_basket_size)))
            while len(basket) < target_size:
                if random.random() < noise_level:
                    basket.add(random.choice(cls.DEFAULT_VOCAB))
                else:
                    # Zipfian pick: lower index items are more popular
                    idx = min(len(cls.DEFAULT_VOCAB) - 1, int(np.random.zipf(1.6)) - 1)
                    basket.add(cls.DEFAULT_VOCAB[idx])

            transactions.append(basket)

        metadata = {
            "source": "synthetic_generator",
            "type": "synthetic_zipfian_affinities",
            "generated_transactions": num_transactions,
            "target_avg_basket": avg_basket_size,
            "noise_level": noise_level
        }
        return TransactionDataset(transactions=transactions, metadata=metadata)
