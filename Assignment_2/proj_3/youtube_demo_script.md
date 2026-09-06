# YouTube E2E Demo Script: CRISP-DM Autoresearch Clustering

**[0:00] Intro & Context**
> *"Hi everyone, today I'll be demonstrating a full-stack Data Mining system built on the CRISP-DM lifecycle, analyzing the Kaggle Customer Segmentation dataset. We'll look at the codebase that drives it and see how it visually maps to our Next.js dashboard."*

---

### Core Part 1: The Multi-Paradigm Model Suite
**[Code Focus]** Show `backend/src/crisp_dm/models/` (specifically `partitioning.py` or `density.py`).
> *"First, let's look at the core modeling pipeline. Instead of relying on a single algorithm, the code abstracts the clustering logic into 4 paradigms: Partitioning, Density, Hierarchical, and Probabilistic. Each algorithm (like HDBSCAN or GMM) implements a common interface, allowing us to swap them out dynamically."*

**[UI Focus]** Navigate to the **Cluster Explorer** view in the UI.
> *"In the UI, this translates directly to the Multi-Model Cluster Explorer. When I select a different algorithm from this dropdown, the backend recalculates the embeddings. You can instantly see the differences in the interactive 3D PCA/UMAP projections and the Silhouette Ribbon plots that determine how well-separated our customer segments are."*

---

### Core Part 2: The Autoresearch Hill-Climbing Engine
**[Code Focus]** Show `backend/src/autoresearch/hill_climber.py` & `objective.py`.
> *"The second core component is the Autoresearch Engine. Finding the right hyperparameters manually is tedious. This code implements a stochastic hill-climbing loop. It automatically mutates preprocessing steps and hyperparameter bounds, then evaluates the result using a custom multi-objective fitness function that balances separation (Silhouette/Davies-Bouldin), stability, and noise."*

**[UI Focus]** Navigate to the **Autoresearch Studio** view in the UI.
> *"Visually, we can monitor this engine live in the Autoresearch Studio. As you can see on this trajectory chart, the engine tracks the best composite fitness score over time. We can watch the parameter deltas shift as the algorithm accepts better configurations and rejects worse ones—effectively automating the experimentation phase of a Data Scientist."*

---

### Core Part 3: Real-Time Inference Scoring
**[Code Focus]** Show `backend/src/api/routes/inference.py`.
> *"Finally, deploying a model requires scoring new data. The FastAPI backend exposes an inference route that accepts raw customer vectors, applies the exact same fitted preprocessing pipeline (like imputation and Yeo-Johnson scaling) from our winning model, and returns soft cluster membership probabilities."*

**[UI Focus]** Navigate to the **Inference Playground** view in the UI.
> *"In the dashboard's Inference Playground, we expose this API to the end-user. As I drag these 17 feature sliders—representing behaviors like 'Credit Limit' or 'Purchases'—the UI instantly hits the backend. The radar chart updates, and we get live predictions classifying this customer into one of our discovered personas, along with actionable business recommendations."*

---

**[Outro]**
> *"By combining a robust Python backend with a reactive TypeScript frontend, we've operationalized the entire CRISP-DM lifecycle. Thanks for watching."*
