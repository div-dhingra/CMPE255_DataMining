"""Synthetic Kaggle Credit Card Dataset Generator for E2E Testing.

Authoritative Schema (18 attributes):
1. CUST_ID: Identification of Credit Card holder (Categorical/String)
2. BALANCE: Balance amount left in their account to make purchases
3. BALANCE_FREQUENCY: How frequently the Balance is updated (score between 0 and 1)
4. PURCHASES: Amount of purchases made from account
5. ONEOFF_PURCHASES: Maximum purchase amount done in one-go
6. INSTALLMENTS_PURCHASES: Amount of purchase done in installment
7. CASH_ADVANCE: Cash in advance given by the user
8. PURCHASES_FREQUENCY: How frequently the Purchases are being made (0 to 1)
9. ONEOFF_PURCHASES_FREQUENCY: How frequently Purchases are happening in one go (0 to 1)
10. PURCHASES_INSTALLMENTS_FREQUENCY: How frequently purchases in installments are being done (0 to 1)
11. CASH_ADVANCE_FREQUENCY: How frequently the cash in advance being paid (0 to 1)
12. CASH_ADVANCE_TRX: Number of Transactions made with "Cash in Advance"
13. PURCHASES_TRX: Number of purchase transactions made
14. CREDIT_LIMIT: Limit of Credit Card for user
15. PAYMENTS: Amount of Payment done by user
16. MINIMUM_PAYMENTS: Minimum amount of payments made by user
17. PRC_FULL_PAYMENT: Percent of full payment paid by user (0 to 1)
18. TENURE: Tenure of credit card service for user (6 to 12)

Archetypes modeled:
- Transactors (Active Spenders): High purchases_freq, high one-off, low cash advance, high prc_full_payment
- Revolvers (Interest Payers): High balance, low purchase freq, low prc_full_payment, moderate cash advance
- Cash Advance Users: High cash_advance_freq, high cash_advance, high balance, low purchases
- Inactive / Low Spenders: Low balance, low purchases, low payments, near-zero transactions
- VIP / High-Limit Spenders: High credit_limit, high purchases, high payments
"""

import csv
import io
import math
import random
from typing import Any, Dict, List, Optional, Tuple, Union


SCHEMA_COLUMNS = [
    "CUST_ID",
    "BALANCE",
    "BALANCE_FREQUENCY",
    "PURCHASES",
    "ONEOFF_PURCHASES",
    "INSTALLMENTS_PURCHASES",
    "CASH_ADVANCE",
    "PURCHASES_FREQUENCY",
    "ONEOFF_PURCHASES_FREQUENCY",
    "PURCHASES_INSTALLMENTS_FREQUENCY",
    "CASH_ADVANCE_FREQUENCY",
    "CASH_ADVANCE_TRX",
    "PURCHASES_TRX",
    "CREDIT_LIMIT",
    "PAYMENTS",
    "MINIMUM_PAYMENTS",
    "PRC_FULL_PAYMENT",
    "TENURE",
]

NUMERICAL_FEATURES = [col for col in SCHEMA_COLUMNS if col != "CUST_ID"]


class SyntheticKaggleDatasetGenerator:
    """Deterministic, configurable generator for synthetic Kaggle Credit Card data."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)

    def reset_seed(self, seed: Optional[int] = None) -> None:
        """Reset internal random generator state."""
        if seed is not None:
            self.seed = seed
        self.rng = random.Random(self.seed)

    def _sample_archetype(self, archetype: str) -> Dict[str, float]:
        """Generate a single customer vector conditioned on archetype."""
        r = self.rng
        if archetype == "transactor":
            balance = max(50.0, r.gauss(800.0, 300.0))
            bal_freq = min(1.0, max(0.2, r.gauss(0.85, 0.1)))
            purchases = max(200.0, r.gauss(2500.0, 800.0))
            oneoff = purchases * r.uniform(0.5, 0.9)
            installments = purchases - oneoff
            cash_adv = max(0.0, r.gauss(50.0, 50.0)) if r.random() < 0.15 else 0.0
            pur_freq = min(1.0, max(0.4, r.gauss(0.85, 0.12)))
            oneoff_freq = pur_freq * r.uniform(0.5, 0.9)
            inst_freq = pur_freq * r.uniform(0.3, 0.7)
            cash_adv_freq = 0.05 if cash_adv > 0 else 0.0
            cash_adv_trx = int(cash_adv / 100.0) if cash_adv > 0 else 0
            pur_trx = max(5, int(r.gauss(30.0, 10.0)))
            credit_limit = max(1000.0, r.gauss(6000.0, 2000.0))
            payments = purchases * r.uniform(0.8, 1.2)
            min_payments = max(50.0, balance * 0.05)
            prc_full = min(1.0, max(0.4, r.gauss(0.75, 0.2)))
            tenure = r.choice([10, 11, 12, 12, 12])
        elif archetype == "revolver":
            balance = max(1000.0, r.gauss(3500.0, 1200.0))
            bal_freq = min(1.0, max(0.7, r.gauss(0.95, 0.05)))
            purchases = max(0.0, r.gauss(400.0, 250.0))
            oneoff = purchases * r.uniform(0.2, 0.6)
            installments = purchases - oneoff
            cash_adv = max(0.0, r.gauss(600.0, 400.0))
            pur_freq = min(1.0, max(0.0, r.gauss(0.25, 0.15)))
            oneoff_freq = pur_freq * 0.5
            inst_freq = pur_freq * 0.5
            cash_adv_freq = min(1.0, max(0.0, r.gauss(0.3, 0.15)))
            cash_adv_trx = max(1, int(r.gauss(4.0, 2.0))) if cash_adv > 0 else 0
            pur_trx = max(1, int(r.gauss(6.0, 4.0)))
            credit_limit = max(1500.0, r.gauss(4500.0, 1500.0))
            payments = max(100.0, balance * r.uniform(0.05, 0.15))
            min_payments = max(80.0, balance * 0.03)
            prc_full = min(0.3, max(0.0, r.gauss(0.02, 0.03)))
            tenure = r.choice([8, 10, 12, 12])
        elif archetype == "cash_advance":
            balance = max(1500.0, r.gauss(4500.0, 1800.0))
            bal_freq = min(1.0, max(0.8, r.gauss(0.98, 0.03)))
            purchases = max(0.0, r.gauss(150.0, 100.0))
            oneoff = purchases * 0.3
            installments = purchases * 0.7
            cash_adv = max(500.0, r.gauss(4000.0, 1500.0))
            pur_freq = min(0.4, max(0.0, r.gauss(0.1, 0.08)))
            oneoff_freq = pur_freq * 0.3
            inst_freq = pur_freq * 0.7
            cash_adv_freq = min(1.0, max(0.3, r.gauss(0.65, 0.15)))
            cash_adv_trx = max(2, int(r.gauss(12.0, 5.0)))
            pur_trx = max(0, int(r.gauss(2.0, 2.0)))
            credit_limit = max(1500.0, r.gauss(5000.0, 1800.0))
            payments = max(200.0, cash_adv * r.uniform(0.3, 0.8))
            min_payments = max(150.0, balance * 0.04)
            prc_full = min(0.1, max(0.0, r.gauss(0.01, 0.02)))
            tenure = r.choice([6, 9, 12, 12])
        elif archetype == "inactive":
            balance = max(0.0, r.gauss(150.0, 100.0))
            bal_freq = min(1.0, max(0.0, r.gauss(0.3, 0.2)))
            purchases = max(0.0, r.gauss(30.0, 30.0))
            oneoff = purchases
            installments = 0.0
            cash_adv = 0.0
            pur_freq = min(0.2, max(0.0, r.gauss(0.05, 0.05)))
            oneoff_freq = pur_freq
            inst_freq = 0.0
            cash_adv_freq = 0.0
            cash_adv_trx = 0
            pur_trx = 1 if purchases > 0 else 0
            credit_limit = max(500.0, r.gauss(2000.0, 800.0))
            payments = purchases + r.uniform(0.0, 50.0)
            min_payments = max(10.0, balance * 0.02)
            prc_full = 0.0 if balance > 0 else 1.0
            tenure = r.choice([6, 7, 8, 12])
        else:  # vip_spender
            balance = max(1000.0, r.gauss(5000.0, 2000.0))
            bal_freq = 1.0
            purchases = max(3000.0, r.gauss(9000.0, 3500.0))
            oneoff = purchases * r.uniform(0.6, 0.9)
            installments = purchases - oneoff
            cash_adv = max(0.0, r.gauss(500.0, 500.0)) if r.random() < 0.2 else 0.0
            pur_freq = min(1.0, max(0.7, r.gauss(0.95, 0.05)))
            oneoff_freq = min(1.0, max(0.5, r.gauss(0.85, 0.1)))
            inst_freq = min(1.0, max(0.4, r.gauss(0.75, 0.15)))
            cash_adv_freq = 0.1 if cash_adv > 0 else 0.0
            cash_adv_trx = int(cash_adv / 250.0) if cash_adv > 0 else 0
            pur_trx = max(20, int(r.gauss(75.0, 25.0)))
            credit_limit = max(8000.0, r.gauss(16000.0, 4000.0))
            payments = purchases * r.uniform(0.9, 1.4)
            min_payments = max(200.0, balance * 0.05)
            prc_full = min(1.0, max(0.3, r.gauss(0.6, 0.25)))
            tenure = 12

        return {
            "BALANCE": round(balance, 2),
            "BALANCE_FREQUENCY": round(bal_freq, 4),
            "PURCHASES": round(purchases, 2),
            "ONEOFF_PURCHASES": round(oneoff, 2),
            "INSTALLMENTS_PURCHASES": round(installments, 2),
            "CASH_ADVANCE": round(cash_adv, 2),
            "PURCHASES_FREQUENCY": round(pur_freq, 4),
            "ONEOFF_PURCHASES_FREQUENCY": round(oneoff_freq, 4),
            "PURCHASES_INSTALLMENTS_FREQUENCY": round(inst_freq, 4),
            "CASH_ADVANCE_FREQUENCY": round(cash_adv_freq, 4),
            "CASH_ADVANCE_TRX": int(cash_adv_trx),
            "PURCHASES_TRX": int(pur_trx),
            "CREDIT_LIMIT": round(credit_limit, 2),
            "PAYMENTS": round(payments, 2),
            "MINIMUM_PAYMENTS": round(min_payments, 2),
            "PRC_FULL_PAYMENT": round(prc_full, 4),
            "TENURE": int(tenure),
        }

    def generate_records(
        self,
        n_rows: int = 500,
        missing_rate_min_payments: float = 0.035,
        missing_rate_credit_limit: float = 0.001,
        outlier_rate: float = 0.02,
        id_prefix: str = "C",
        start_id: int = 10001,
    ) -> List[Dict[str, Any]]:
        """Generate a list of records with specified size and characteristics."""
        archetypes = ["transactor", "revolver", "cash_advance", "inactive", "vip_spender"]
        weights = [0.30, 0.25, 0.20, 0.15, 0.10]

        records: List[Dict[str, Any]] = []
        for i in range(n_rows):
            arch = self.rng.choices(archetypes, weights=weights, k=1)[0]
            row_dict = self._sample_archetype(arch)
            cust_id = f"{id_prefix}{start_id + i}"
            full_row: Dict[str, Any] = {"CUST_ID": cust_id}
            full_row.update(row_dict)

            # Inject outliers if selected
            if self.rng.random() < outlier_rate:
                feature_to_spike = self.rng.choice(["BALANCE", "PURCHASES", "CASH_ADVANCE", "PAYMENTS"])
                full_row[feature_to_spike] = round(full_row[feature_to_spike] * self.rng.uniform(5.0, 15.0), 2)

            # Inject missing values
            if self.rng.random() < missing_rate_min_payments:
                full_row["MINIMUM_PAYMENTS"] = None
            if self.rng.random() < missing_rate_credit_limit:
                full_row["CREDIT_LIMIT"] = None

            records.append(full_row)

        return records

    def generate_csv_string(
        self,
        n_rows: int = 500,
        missing_rate_min_payments: float = 0.035,
        missing_rate_credit_limit: float = 0.001,
    ) -> str:
        """Generate CSV formatted string of synthetic dataset."""
        records = self.generate_records(
            n_rows=n_rows,
            missing_rate_min_payments=missing_rate_min_payments,
            missing_rate_credit_limit=missing_rate_credit_limit,
        )
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=SCHEMA_COLUMNS)
        writer.writeheader()
        for r in records:
            writer.writerow(r)
        return output.getvalue()

    def generate_matrix(
        self,
        n_rows: int = 100,
        impute_strategy: str = "median",
    ) -> Tuple[List[List[float]], List[str]]:
        """Return a 2D numerical matrix (N x 17) with imputed missing values."""
        records = self.generate_records(n_rows=n_rows)
        # Compute median/mean for imputation
        min_payments_vals = [r["MINIMUM_PAYMENTS"] for r in records if r["MINIMUM_PAYMENTS"] is not None]
        credit_limit_vals = [r["CREDIT_LIMIT"] for r in records if r["CREDIT_LIMIT"] is not None]

        def get_stat(vals: List[float], strategy: str) -> float:
            if not vals:
                return 0.0
            if strategy == "mean":
                return sum(vals) / len(vals)
            s = sorted(vals)
            return s[len(s) // 2]

        imp_min_pay = get_stat(min_payments_vals, impute_strategy)
        imp_cred_lim = get_stat(credit_limit_vals, impute_strategy)

        matrix: List[List[float]] = []
        cust_ids: List[str] = []
        for r in records:
            cust_ids.append(r["CUST_ID"])
            row_vals: List[float] = []
            for col in NUMERICAL_FEATURES:
                val = r[col]
                if val is None:
                    if col == "MINIMUM_PAYMENTS":
                        val = imp_min_pay
                    elif col == "CREDIT_LIMIT":
                        val = imp_cred_lim
                    else:
                        val = 0.0
                row_vals.append(float(val))
            matrix.append(row_vals)

        return matrix, cust_ids


def create_synthetic_dataset(
    n_rows: int = 500,
    seed: int = 42,
    missing_rate_min_payments: float = 0.035,
    missing_rate_credit_limit: float = 0.001,
) -> str:
    """Convenience helper to generate synthetic CSV data."""
    gen = SyntheticKaggleDatasetGenerator(seed=seed)
    return gen.generate_csv_string(
        n_rows=n_rows,
        missing_rate_min_payments=missing_rate_min_payments,
        missing_rate_credit_limit=missing_rate_credit_limit,
    )
