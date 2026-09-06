# CRISP-DM Process Documentation: Associative Pattern Mining System

This document outlines the end-to-end execution of the **CRISP-DM (Cross-Industry Standard Process for Data Mining)** methodology for the Associative Pattern Mining & Autoresearch System.

```
       ┌─────────────────────────────────────────────────────────┐
       │              1. BUSINESS UNDERSTANDING                  │
       │   - Retail Basket Affinity & Cross-Sell Uplift          │
       │   - Merchandising & Bundle Pricing Strategies           │
       └──────────────────────────┬──────────────────────────────┘
                                  │
                                  ▼
       ┌─────────────────────────────────────────────────────────┐
       │               2. DATA UNDERSTANDING                     │
       │   - Kaggle Online Retail / Transactional Logs           │
       │   - Power-law (Zipfian) Item Frequencies & Sparsity     │
       └──────────────────────────┬──────────────────────────────┘
                                  │
                                  ▼
       ┌─────────────────────────────────────────────────────────┐
       │                3. DATA PREPARATION                      │
       │   - Return/Cancellation Scrubbing (InvoiceNo 'C')       │
       │   - Basket Aggregation & One-Hot Matrix Encoding        │
       │   - Vertical TID Bitset Conversion for Fast Joins       │
       └──────────────────────────┬──────────────────────────────┘
                                  │
                                  ▼
       ┌─────────────────────────────────────────────────────────┐
       │                     4. MODELING                         │
       │   - Apriori (Downward-Closure Candidate Pruning)        │
       │   - FP-Growth (Frequent Pattern Tree & F-List)          │
       │   - ECLAT (Equivalence Class Lattice DFS Intersections) │
       └──────────────────────────┬──────────────────────────────┘
                                  │
                                  ▼
       ┌─────────────────────────────────────────────────────────┐
       │                    5. EVALUATION                        │
       │   - 7 Metrics: Supp, Conf, Lift, Conv, Lev, Zhang, Kulc │
       │   - Null-Invariance & Imbalance Ratio (IR) Analysis     │
       │   - Parsimonious Non-Redundant Rule Pruning             │
       └──────────────────────────┬──────────────────────────────┘
                                  │
                                  ▼
       ┌─────────────────────────────────────────────────────────┐
       │                    6. DEPLOYMENT                        │
       │   - FastAPI High-Throughput REST & SSE Service          │
       │   - Real-Time Cart Recommendation Playground            │
       │   - Autonomous Hill-Climbing Autoresearch Optimizer     │
       └─────────────────────────────────────────────────────────┘
```

---

## Phase 1: Business Understanding

### 1.1 Business Context & Problem Statement
In modern e-commerce and brick-and-mortar retail environments, understanding customer co-purchasing behavior is essential for driving revenue growth. Traditional merchandising relies on merchant intuition, which fails to scale across catalogs with thousands of stock-keeping units (SKUs) and rapidly shifting consumer tastes.

Associative pattern mining uncovers latent dependencies between items:
$$\text{Antecedent } (A) \implies \text{Consequent } (C)$$

### 1.2 Strategic Business Objectives
1. **Maximize Average Order Value (AOV)**: Identify high-affinity item bundles (e.g. teapot + teacup saucer) that can be dynamically suggested at checkout.
2. **Promotional Cross-Sell Uplift**: Determine which anchor products can drive sales of higher-margin complementary goods without cannibalizing baseline demand.
3. **Optimized Shelf / Web Placement**: Position complementary products in close proximity (physical shelves or online recommended add-on sections) to minimize customer search friction.
4. **Autonomous Parameter Optimization**: Eliminate manual guessing of mining thresholds (`min_support`, `min_confidence`, `min_lift`) via a closed-loop autoresearch engine.

### 1.3 Target Success KPIs
- **Precision / Rule Confidence**: Minimum 40% confidence threshold for automated recommendations to prevent spamming irrelevant items.
- **Correlation Lift**: Minimum $1.15\times$ lift to filter out coincidental co-occurrences of independently ubiquitous items.
- **Serving Latency**: $< 10\text{ ms}$ inference latency for real-time cart recommendations.

---

## Phase 2: Data Understanding

### 2.1 Dataset Ingestion
The system ingests transactional retail databases, supporting both the classic **Kaggle Online Retail Dataset** (UK gift retailer) and controlled synthetic benchmarks:
- **InvoiceNo**: Unique 6-digit transaction identifier. Invoices starting with `'C'` denote cancellations.
- **StockCode**: Unique product identifier.
- **Description**: Human-readable product name.
- **Quantity**: Units purchased per transaction.
- **UnitPrice**: Product price in Sterling.
- **CustomerID**: Unique 5-digit identifier for customer accounts.
- **Country**: Geographic origin of purchase.

### 2.2 Data Characteristics & Sparsity
Transactional databases present unique challenges distinct from traditional regression or classification tabular data:
1. **Extreme Sparsity**: With thousands of distinct SKUs, an individual customer purchase contains on average 3 to 6 items. The transaction-item matrix density is typically $< 0.5\%$.
2. **Zipfian / Power-Law Distribution**: A tiny fraction of catalog items (e.g., shopping bags, white hanging heart candle holders) account for a disproportionately large volume of sales, while the long tail contains thousands of low-frequency items.
3. **Null-Transaction Distortion**: The overwhelming majority of transactions contain neither item $A$ nor item $C$. Standard symmetric measures (e.g., Pearson correlation, $\chi^2$) are distorted by the massive count of negative co-occurrences ($f_{00}$), necessitating **null-invariant metrics** (Tan et al. 2004).

---

## Phase 3: Data Preparation

### 3.1 Data Cleaning & Filtering Pipeline
1. **Cancellation & Return Removal**: Records with `InvoiceNo` prefixed with `'C'` or non-positive `Quantity <= 0` are filtered out to prevent distorted negative affinities.
2. **SKU Sanitization**: Administrative and non-merchandise line items (e.g. `"POSTAGE"`, `"Manual"`, `"DOTCOM POSTAGE"`, `"Adjust bad debt"`) are removed.
3. **Item Standardization**: Whitespace trimming, uppercase normalization, and length filtering ($> 1$ character).

### 3.2 Transaction Aggregation & Encoding
The cleaned tabular rows are aggregated by `InvoiceNo` to form transaction sets:
$$\mathcal{D} = \{T_1, T_2, \dots, T_N\}, \quad T_i \subseteq \mathcal{I}$$
Where $\mathcal{I}$ is the set of all unique catalog items.

Data representations created:
1. **List of Frozensets**: In-memory representation for rapid $O(1)$ item lookup and subset checks (`A.issubset(T)`).
2. **One-Hot Boolean DataFrame**: Sparse 2D matrix where rows represent transactions and columns represent items ($M_{i, j} \in \{0, 1\}$).
3. **Vertical Database Layout (Inverted Index)**: Mapping each item to its set of Transaction IDs (TIDs):
   $$\text{item } i \mapsto \{tid_1, tid_4, tid_9, \dots\}$$
   Enabling pure bitset intersection operations for the ECLAT algorithm.

---

## Phase 4: Modeling

The system implements three distinct algorithmic paradigms from scratch in pure Python / NumPy, enabling empirical benchmarking against theoretical literature:

### 4.1 Apriori (Agrawal & Srikant, VLDB 1994)
- **Paradigm**: Breadth-first level-wise search with candidate generation.
- **Core Mechanism**: Leverages the downward closure property of support (anti-monotonicity):
  $$\forall X, Y: X \subseteq Y \implies \text{supp}(Y) \le \text{supp}(X)$$
  If an itemset is infrequent, all of its supersets are guaranteed to be infrequent and are pruned immediately.
- **Candidate Join & Pruning**:
  1. $L_{k-1} \Join L_{k-1}$: Two frequent $(k-1)$-itemsets sharing $(k-2)$ items are joined into candidate $k$-itemset.
  2. Pruning: Verify that all $(k-1)$-subsets of the candidate exist in $L_{k-1}$.
  3. Database Scan: Scan transactions to count occurrences of remaining candidates.
- **Bottleneck**: Low `min_support` values cause candidate explosion ($|C_k|$) and require $k$ full transactional database scans.

### 4.2 FP-Growth (Han, Pei, & Yin, ACM SIGMOD 2000)
- **Paradigm**: Frequent Pattern Tree (FP-Tree) divide-and-conquer without candidate generation.
- **Core Mechanism**:
  1. **F-List Construction**: Scan database once to find frequent 1-itemsets; sort items in descending support order.
  2. **Tree Insertion**: Scan database second time; insert items of each transaction in F-List order into a prefix tree. Nodes with shared prefixes are merged and counts incremented.
  3. **Header Table & Node Links**: A header table links all tree nodes containing the same item across branches.
  4. **Conditional FP-Tree Mining**: Starting from the least frequent items in the header table, trace parent paths to construct the conditional pattern base, derive conditional trees, and recurse.
- **Advantage**: Requires only 2 database passes and eliminates candidate joins entirely.

### 4.3 ECLAT (Zaki, IEEE TKDE 2000)
- **Paradigm**: Equivalence Class Clustering and bottom-up Lattice Traversal using vertical database layout.
- **Core Mechanism**:
  - Encodes transactions vertically: each item is mapped to its TID set.
  - Support of $X \cup Y$ is computed via set intersection:
    $$\text{TID}(X \cup Y) = \text{TID}(X) \cap \text{TID}(Y)$$
    $$\text{supp}(X \cup Y) = \frac{|\text{TID}(X) \cap \text{TID}(Y)|}{N}$$
  - Traverses the itemset lattice in Depth-First Search (DFS) order.
- **Advantage**: Eliminates horizontal database scanning; fast set intersections on sparse datasets.

---

## Phase 5: Evaluation

### 5.1 Comprehensive Interestingness Metrics (7 Metrics)

| Metric | Formula | Range | Interpretation & Theoretical Significance |
|---|---|---|---|
| **Support** | $P(A \cup C) = \frac{\sigma(A \cup C)}{N}$ | $[0, 1]$ | Baseline probability that both antecedent and consequent appear in a transaction. |
| **Confidence** | $\frac{P(A \cup C)}{P(A)}$ | $[0, 1]$ | Conditional probability $P(C \mid A)$. Measures rule reliability. |
| **Lift** | $\frac{P(A \cup C)}{P(A) \cdot P(C)}$ | $[0, \infty)$ | Ratio of observed co-occurrence to expected under independence. $>1$ indicates positive correlation. |
| **Conviction** | $\frac{1 - P(C)}{1 - \text{conf}(A \to C)}$ | $[0, \infty)$ | Ratio of expected frequency of $A$ without $C$ under independence to observed failure frequency. |
| **Leverage** | $P(A \cup C) - P(A)P(C)$ | $[-0.25, 0.25]$ | Difference between observed and expected co-occurrence. Measures absolute surplus transactions. |
| **Zhang's Metric** | $\frac{D}{\max(P(A \cup C)(1-P(A)), P(A)(P(C)-P(A \cup C)))}$ | $[-1, 1]$ | Normalized association measure distinguishing positive association ($>0$), independence ($0$), and negative association ($<0$). |
| **Kulczynski (Kulc)** | $\frac{1}{2}\left(\frac{P(A \cup C)}{P(A)} + \frac{P(A \cup C)}{P(C)}\right)$ | $[0, 1]$ | **Null-invariant** arithmetic mean of directional confidences. Neutrality at $0.5$, positive affinity $>0.5$. |
| **Imbalance Ratio (IR)** | $\frac{\|P(A) - P(C)\|}{P(A) + P(C) - P(A \cup C)}$ | $[0, 1]$ | Quantifies transaction frequency asymmetry between antecedent and consequent. |

### 5.2 Redundant Rule Elimination
A discovered rule $A' \implies C$ is classified as **redundant** if there exists another rule $A \implies C$ with $A \subset A'$ such that:
$$\text{conf}(A \implies C) \ge \text{conf}(A' \implies C)$$
The shorter, more general antecedent $A$ is strictly more parsimonious and equally or more reliable. The system prunes these redundant rules automatically.

---

## Phase 6: Deployment & Continuous Optimization

### 6.1 Serving Architecture
1. **High-Performance FastAPI Backend**: REST endpoints serving dataset statistics, rule mining, interactive graph representations, and real-time cart recommendations.
2. **Interactive Data Science Admin Dashboard**:
   - 6-phase CRISP-DM lifecycle tracker and dataset audit.
   - Force-directed **Vis.js Rule Network Graph** with physics simulation, node ego-networks, and edge drilldowns.
   - **Rules Matrix Scatter Plot** (Support vs. Confidence vs. Lift).
   - **Autoresearch Studio** with live convergence curves and experiment leaderboard.
   - **Market Basket Recommendation Sandbox** for interactive cart testing.

### 6.2 Autonomous Hill-Climbing Optimization Engine
The autoresearch engine searches the hyperparameter space $\Theta = (\text{supp}, \text{conf}, \text{lift}, \text{alg}, \text{len})$ to discover optimal mining configurations without human trial-and-error:
- **Composite Multi-Objective Fitness**:
  $$F(\theta) = 0.30 \cdot C_{cov} + 0.35 \cdot I_{int} + 0.25 \cdot D_{div} - 0.10 \cdot R_{pen} - S_{pen}$$
- **Simulated Annealing Acceptance**:
  $$P(\text{accept}) = \begin{cases} 1.0 & \text{if } \Delta F > 0 \\ \exp\left(\frac{\Delta F}{T_k}\right) & \text{if } \Delta F \le 0 \end{cases}$$
- **Tabu Cache**: Prevents cycling back to recently visited parameter states.
- **Random Restarts**: Automatically triggered after stagnation to escape local optima.
