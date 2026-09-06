# Agent Skills Demonstration Platform - Video Demo Script

## 1. Introduction (0:00 - 0:15)
**Visual:** Screen recording starts on the Dashboard UI homepage (Tab 1: CRISP-DM Lifecycle).
**Voiceover:** "Welcome to the Agent Skills CRISP-DM Demonstration Platform. Today we're looking at an end-to-end Machine Learning and Data Analytics pipeline built to predict customer churn, integrating 46 production-grade agent skills. Let's look at how the core code powers this interactive dashboard."

## 2. Core Code to UI Mapping (0:15 - 1:15)

### Part 1: Phase 3 - Data Preparation & Handling Imbalance
**Visual (Code):** Switch to your IDE. Open `src/modules/preprocessor.py` and highlight `clean_dataset_leakage_safe()` and `handle_imbalanced_data()`.
**Voiceover:** "First, we ensure strict data integrity. In `src/modules/preprocessor.py`, we implement a zero-leakage preprocessing pipeline and a synthetic minority oversampling (SMOTE) logic to balance our dataset."
**Visual (Dashboard):** Switch back to the UI. Click on **Tab 4: Executive KPI & Observability** and point to the "Dataset Distribution: Churn Class Imbalance" figure, or scroll through the Phase 3 pipeline execution in Tab 1.
**Voiceover:** "This translates directly into the application's reliable baseline. In the dashboard, you can see how our class imbalance is handled before model training even begins, preventing skewed predictions."

### Part 2: Phase 4 - Model Training & Cross-Validation
**Visual (Code):** Switch to IDE. Open `src/modules/model_pipeline.py` and highlight the `build_model_pipelines()` and `execute_cross_validation()` functions.
**Voiceover:** "Next, in `src/modules/model_pipeline.py`, we build Scikit-Learn pipelines for models like Random Forest and HistGradientBoosting, tuning them using stratified 5-fold cross-validation."
**Visual (Dashboard):** Switch to the UI. Go to **Tab 1: CRISP-DM Lifecycle** and highlight the Top KPI Card: **Champion Model ROC-AUC (0.8504)**. 
**Voiceover:** "The outputs of this rigorous tuning process are dynamically served to the UI, surfacing our champion model's exact ROC-AUC score and PR-AUC metrics right in the executive summary."

### Part 3: Phase 6 - Real-Time Inference Router
**Visual (Code):** Switch to IDE. Open `src/server/routes/inference_router.py` and highlight the `predict_churn()` endpoint, specifically the logic generating `risk_tier` and `retention_offer`.
**Voiceover:** "Finally, to make these insights actionable, `src/server/routes/inference_router.py` hosts a FastAPI POST endpoint that ingests live customer profiles, scores their churn probability, and calculates a targeted retention offer based on their risk tier."
**Visual (Dashboard):** Switch to the UI. Click on **Tab 3: Real-Time Churn Profiler**. Load the 'High-Risk New Fiber Subscriber' preset and click 'Predict Customer Churn Risk'.
**Voiceover:** "This directly powers the Real-Time Churn Profiler tab. When we adjust customer attributes and run the simulator, it hits this endpoint to instantly return their churn probability, top risk drivers, and a tailored proactive retention intervention."

## 3. Conclusion (1:15 - 1:30)
**Visual:** Scroll through the Skills Catalog (Tab 2) briefly.
**Voiceover:** "By seamlessly combining robust ML backend modules with an interactive frontend, this platform demonstrates the practical impact of deploying standardized data and ML agent skills. Thanks for watching."
