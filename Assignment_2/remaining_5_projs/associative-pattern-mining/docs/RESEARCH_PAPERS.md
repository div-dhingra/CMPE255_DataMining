# Theoretical Foundations & Academic Literature Alignment

This document details the research papers and theoretical principles implemented within the Associative Pattern Mining & Autoresearch System, highlighting the mathematical formulations, algorithmic paradigms, and empirical trade-offs.

---

## 1. Agrawal & Srikant (VLDB 1994)
**"Fast Algorithms for Mining Association Rules"**  
*Rakesh Agrawal and Ramakrishnan Srikant*  
*Proceedings of the 20th International Conference on Very Large Data Bases (VLDB), 1994, pp. 487–499.*

### Core Contributions & Theorems
- **The Apriori Property (Anti-Monotonicity / Downward Closure of Support)**:
  $$\forall X, Y \subseteq \mathcal{I}: \quad X \subseteq Y \implies \text{supp}(Y) \le \text{supp}(X)$$
  **Theorem**: If an itemset $X$ is infrequent ($\text{supp}(X) < \text{minsupp}$), then no superset $Y \supseteq X$ can be frequent.
- **Candidate Generation ($L_{k-1} \Join L_{k-1}$)**:
  A candidate $k$-itemset $C_k$ is constructed by joining two frequent $(k-1)$-itemsets $p, q \in L_{k-1}$ if they share the first $k-2$ items in lexicographical order:
  $$p = \{i_1, i_2, \dots, i_{k-2}, i_{k-1}\}, \quad q = \{i_1, i_2, \dots, i_{k-2}, i'_{k-1}\}, \quad i_{k-1} < i'_{k-1}$$
  $$c = p \cup q = \{i_1, i_2, \dots, i_{k-2}, i_{k-1}, i'_{k-1}\}$$
- **Apriori Pruning Step**:
  For candidate $c \in C_k$, if any $(k-1)$-subset $s \subset c$ is not in $L_{k-1}$, $c$ is pruned prior to scanning the database.

### System Implementation & Empirical Trade-Offs
- Implemented in [`backend/src/algorithms/apriori.py`](file:///Users/divdhingra-personal/Desktop/CMPE255_DataMining/Assignment_2/remaining_5_projs/associative-pattern-mining/backend/src/algorithms/apriori.py).
- **Empirical Observation**: Apriori performs exceptionally well when `min_support` is high ($> 5\%$) and itemset lengths are small ($\le 2$). However, as `min_support` decreases, the cardinality $|C_k|$ explodes exponentially, requiring $k$ complete scans of the transaction database.

---

## 2. Han, Pei, & Yin (ACM SIGMOD 2000)
**"Mining Frequent Patterns without Candidate Generation: A Frequent-Pattern Tree Approach"**  
*Jiawei Han, Jian Pei, and Yiwen Yin*  
*ACM SIGMOD Record, Vol. 29, No. 2, 2000, pp. 1–12.*

### Core Contributions & Architectural Innovations
- **FP-Tree (Frequent Pattern Tree)**:
  A compact, prefix-tree data structure that stores compressed transactional data:
  1. Root labeled `null`.
  2. Subtree nodes labeled with `item`, `count`, `parent` pointer, and `children` dictionary.
  3. **Header Table**: Contains the list of frequent items sorted by frequency (F-List) with linked-list head pointers (`node_link`) traversing all tree nodes for that item across different branches.
- **Divide-and-Conquer Recursive Mining**:
  Eliminates candidate generation entirely:
  1. For each frequent item $i$ in bottom-up header table order:
     - Form pattern $\text{prefix} \cup \{i\}$.
     - Construct its **Conditional Pattern Base** by following parent pointers from $i$'s linked-list nodes to root.
     - Build a **Conditional FP-Tree** from the accumulated prefix paths.
     - Recurse on the conditional tree.

### System Implementation & Empirical Trade-Offs
- Implemented in [`backend/src/algorithms/fpgrowth.py`](file:///Users/divdhingra-personal/Desktop/CMPE255_DataMining/Assignment_2/remaining_5_projs/associative-pattern-mining/backend/src/algorithms/fpgrowth.py).
- **Empirical Observation**: Requires exactly 2 transactional database scans. On the Kaggle Online Retail dataset, FP-Growth is orders of magnitude faster than Apriori at low support thresholds ($\text{supp} \le 0.02$) because it never generates candidate pairs that do not exist in the transactional data.

---

## 3. Zaki (IEEE TKDE 2000)
**"Scalable Algorithms for Association Mining"**  
*Mohammed J. Zaki*  
*IEEE Transactions on Knowledge and Data Engineering, Vol. 12, No. 3, 2000, pp. 372–390.*

### Core Contributions & Vertical Layout
- **Vertical Database Layout (TID Sets)**:
  Instead of horizontal transactions (TID $\to$ items), represent each item by the set of Transaction IDs in which it occurs:
  $$\mathcal{T}(X) = \{tid \mid X \subseteq T_{tid}\}$$
- **Support by Set Intersection**:
  $$\mathcal{T}(X \cup Y) = \mathcal{T}(X) \cap \mathcal{T}(Y)$$
  $$\text{supp}(X \cup Y) = \frac{|\mathcal{T}(X) \cap \mathcal{T}(Y)|}{N}$$
- **Lattice Equivalence Classes (ECLAT)**:
  Equivalence class clustering partitions the search lattice into prefix equivalence classes, enabling memory-efficient Depth-First Search (DFS) traversals without rescanning the database.

### System Implementation & Empirical Trade-Offs
- Implemented in [`backend/src/algorithms/eclat.py`](file:///Users/divdhingra-personal/Desktop/CMPE255_DataMining/Assignment_2/remaining_5_projs/associative-pattern-mining/backend/src/algorithms/eclat.py).
- **Empirical Observation**: Extremely rapid when transactions are sparse and individual TID set cardinality is low. Support calculation is reduced to simple Python set intersections.

---

## 4. Tan, Kumar, & Srivastava (ACM SIGKDD 2002 / Info. Systems 2004)
**"Selecting the Right Objective Measure for Association Analysis"**  
*Pang-Ning Tan, Vipin Kumar, and Jaideep Srivastava*  
*Information Systems, Vol. 29, No. 4, 2004, pp. 293–331.*

### Key Theoretical Properties of Interestingness Measures
Tan et al. established rigorous mathematical criteria for evaluating objective association measures:
1. **Symmetry Property**: $M(A \to C) = M(C \to A)$. (e.g. Lift is symmetric; Confidence is asymmetric).
2. **Null-Invariance Property**:
   Let the $2 \times 2$ contingency table be:
   - $f_{11} = |A \cap C|$
   - $f_{10} = |A \cap \neg C|$
   - $f_{01} = |\neg A \cap C|$
   - $f_{00} = |\neg A \cap \neg C|$ (Null transactions)
   **Definition**: A measure $M$ is **null-invariant** if its value is independent of $f_{00}$.
   - **Crucial Finding**: In massive retail databases, $f_{00} \gg f_{11}, f_{10}, f_{01}$. Non-null-invariant measures (e.g. $\chi^2$, Cosine correlation, Odds Ratio) are heavily distorted by the sheer volume of empty co-occurrences.
   - **Kulczynski (Kulc)** and **Imbalance Ratio (IR)** are null-invariant and provide the most reliable signals for sparse retail transactions.

---

## 5. Wu, Chen, & Han (DMKD 2007)
**"Association Pruning: Null-Invariance and Association Analysis in Large Transaction Databases"**  
*Tianyi Wu, Yuguo Chen, and Jiawei Han*  
*Data Mining and Knowledge Discovery, Vol. 15, No. 3, 2007.*

### Kulczynski Measure & Imbalance Ratio Formulation
Wu et al. demonstrated that analyzing associations using **Kulczynski (Kulc)** together with **Imbalance Ratio (IR)** resolves classic retail mining paradoxes:
$$\text{Kulc}(A, C) = \frac{1}{2} \left( P(C \mid A) + P(A \mid C) \right) = \frac{1}{2} \left( \frac{P(A \cup C)}{P(A)} + \frac{P(A \cup C)}{P(C)} \right)$$
$$\text{IR}(A, C) = \frac{|P(A) - P(C)|}{P(A) + P(C) - P(A \cup C)}$$
- $\text{Kulc} \in [0, 1]$: $0.5$ represents neutral independence; $> 0.5$ indicates strong positive affinity.
- $\text{IR} \in [0, 1]$: $0$ represents balanced frequency; values approaching $1$ indicate severe imbalance (e.g. one item is a rare luxury good and the other is a daily staple).

Implemented in [`backend/src/core/metrics.py`](file:///Users/divdhingra-personal/Desktop/CMPE255_DataMining/Assignment_2/remaining_5_projs/associative-pattern-mining/backend/src/core/metrics.py).

---

## 6. Sakana AI / Karpathy (2024 Autoresearch Paradigm)
**"The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery"**  
*Sakana AI, 2024.*

### Autonomous Closed-Loop Optimization Loop
Traditional association rule mining requires human data scientists to manually adjust arbitrary cutoffs (`min_support`, `min_confidence`, `min_lift`). This causes either:
- **Rule Starvation**: Thresholds too strict $\implies$ zero or trivial single-item rules found.
- **Rule Explosion**: Thresholds too loose $\implies$ combinatorial explosion of millions of redundant rules, exhausting RAM and crashing downstream microservices.

The **Autoresearch Engine** implements an autonomous closed-loop search:
$$\theta^* = \arg\max_{\theta \in \Theta} F(\theta)$$
With:
1. **Bounded Search Space $\Theta$**: Spanning continuous support, confidence, lift, and discrete algorithms.
2. **Multi-Objective Composite Fitness Function**: Balancing item catalog coverage, rule interestingness, non-redundancy, and execution runtime.
3. **Simulated Annealing Transition Probability**:
   $$P(\text{accept}) = \exp\left(\frac{\Delta F}{T_k}\right)$$
   Allowing escape from local sub-optimal plateaus.
4. **Tabu Cache & Stagnation Restarts**: Guarantees broad exploration of the parameter landscape.
5. **Experiment Ledger**: Complete structured audit trail of all historical iterations, deltas, and convergence steps.
