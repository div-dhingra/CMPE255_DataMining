# Video Demo Script: Associative Pattern Mining

This script provides a concise, end-to-end walkthrough of the Associative Pattern Mining & Autoresearch Studio. It connects three core backend implementations directly to their corresponding visual interactive elements in the frontend dashboard.

---

### Introduction (0:00 - 0:15)
* **Visual:** Show the main dashboard view, specifically highlighting the "CRISP-DM Lifecycle & Audit" tab.
* **Voiceover:** "Welcome to the Associative Pattern Mining Studio. Today, we'll walk through our built-from-scratch frequent itemset mining framework, highlighting how our backend algorithms power real-time frontend analytics across the CRISP-DM lifecycle."

---

### Part 1: Custom FP-Growth Algorithm & Rules Mining (0:15 - 0:45)
* **Code Highlight:** Point to `backend/src/algorithms/fpgrowth.py`.
* **Voiceover:** "To start, we implemented three core mining algorithms from scratch, bypassing heavy ML libraries. Here in `fpgrowth.py`, we construct a Frequent Pattern Tree (`FPNode`) to efficiently mine itemsets without candidate generation."
* **UI Connection:** Switch to the **"Rules Matrix Scatter & Mining" (View 3)** in the dashboard (`#scatter-matrix-view`).
* **Voiceover:** "In the dashboard's Rules Matrix tab, you can select 'FP-Growth', adjust your minimum support and confidence thresholds, and click 'Run Mining Execution'. The backend leverages our custom FP-Growth tree to instantly generate and plot the resulting association rules on the scatter chart below."

---

### Part 2: The 7 Interestingness Metrics (0:45 - 1:15)
* **Code Highlight:** Point to `backend/src/core/metrics.py`.
* **Voiceover:** "Simply finding frequent itemsets isn't enough; we need to measure their significance. In `metrics.py`, we calculate seven distinct interestingness metrics for every generated rule, including standard metrics like Support, Confidence, and Lift, as well as advanced metrics like Kulczynski, Conviction, and Zhang's Metric."
* **UI Connection:** Switch to the **"Association Rule Network Graph" (View 2)** and the **Rules Table** in View 3.
* **Voiceover:** "These calculated metrics are pushed directly to the UI. In the Rules Matrix table, you can sort discovered rules by any of these 7 metrics (e.g., sorting by Kulczynski to handle null-transaction skew). Furthermore, in the Network Graph view, clicking any edge instantly displays these specific metric values for the selected rule."

---

### Part 3: Real-Time Market Basket Recommendations (1:15 - 1:45)
* **Code Highlight:** Point to `backend/src/core/recommendations.py`.
* **Voiceover:** "Finally, to make our mined rules actionable, we implemented a real-time recommendation engine in `recommendations.py`. This script dynamically matches a subset of items (the antecedent) against our active rules to calculate projected cross-sell uplift."
* **UI Connection:** Switch to the **"Market Basket Recommendation Sandbox" (View 5)** (`#cart-simulator-view`).
* **Voiceover:** "We can see this in action in the Recommendation Sandbox tab. As we add products to our active shopping basket from the catalog dropdown, the UI immediately polls the recommendations API. It triggers the matching rules and suggests real-time bundles and add-ons based on our highest-ranking metrics, proving the deployment phase of our CRISP-DM lifecycle."

---

### Conclusion (1:45 - 2:00)
* **Visual:** Quickly tab through the Autoresearch tab and back to the CRISP-DM tab.
* **Voiceover:** "Combined with our autonomous hill-climbing autoresearch engine, this studio provides an end-to-end framework for interactive, algorithm-driven associative pattern mining. Thank you for watching."
