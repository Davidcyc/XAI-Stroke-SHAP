# Reproducibility Appendix & Technical Documentation

This appendix documents the complete experimental environment, data preprocessing pipeline, model architectures, hyperparameter configurations, and SHAP explainer specifications used in the stroke classification and explainability study.

---

## 1. Experimental Environment & Dependency Versions

All experiments were executed under an isolated Conda environment (`xai_env`) on Windows 64-bit with Python 3.10.

| Component / Library | Version | Analytical Role |
| :--- | :--- | :--- |
| **Python** | 3.10.22 | Base execution runtime |
| **pandas** | 2.3.3 | Data ingestion, descriptive statistics, and feature engineering |
| **numpy** | 2.2.6 | Array manipulations, metric computations, and residual evaluations |
| **scikit-learn** | 1.7.2 | Stratified data partitioning, imputation, model training, and metrics |
| **imbalanced-learn** | 0.14.2 | Minority oversampling (`RandomOverSampler`) |
| **shap** | 0.49.1 | TreeExplainer, KernelExplainer, force plots, and beeswarm summary plots |
| **matplotlib** | 3.10.9 | High-resolution publication-quality figure generation (300 DPI) |
| **seaborn** | 0.13.2 | Distributional plots (KDE and comorbidity charts) |
| **python-docx** | 1.2.0 | Programmatic Word template compilation |
| **python-pptx** | 1.0.2 | Programmatic compilation of the three-audience slide deck |

---

## 2. Dataset Provenance & Safeguards Against Data Leakage

### 2.1 Dataset Profile
- **File**: `healthcare-dataset-stroke-data.csv` (located in `xai-series-master/xai-series-master/data/`)
- **Dimensions**: 5,110 patient rows × 12 columns
- **Target Variable**: `stroke` (binary: 0 = no stroke, 1 = recorded stroke history)
- **Class Balance**: 4,861 negative (95.13%) vs. 249 positive (4.87%) — an imbalance ratio of ~19.5 : 1

### 2.2 Data Leakage Safeguards (Strict Protocol)
1. **Identifier Removal**: `id` was dropped before modeling as it has zero predictive relevance.
2. **Stratified Split Before Any Transformation**:
   - `test_size=0.20`, `random_state=42`, `stratify=y`
   - Training split: 4,088 instances (3,889 negative, 199 positive)
   - Held-out test split: 1,022 instances (972 negative, 50 positive)
3. **Training-Only Imputation**:
   - `bmi` had 201 missing values (3.93%).
   - The median of the training split (**28.00 kg/m²**) was computed and used to impute both train and test splits, preventing distribution leakage from the test split.
4. **Encoding & Alignment**:
   - Categorical predictors (`gender`, `ever_married`, `work_type`, `Residence_type`, `smoking_status`) were one-hot encoded.
   - Column names and orderings were strictly aligned across splits, yielding 21 feature columns.
   - `smoking_status == "Unknown"` was preserved as an explicit category to retain informative non-response patterns.
5. **Training-Only Oversampling**:
   - `RandomOverSampler(sampling_strategy='minority', random_state=42)` was fitted **strictly on the training split**, balancing it to 7,778 records (3,889 per class).
   - **The test split was kept completely untouched and reflects the natural 4.89% prevalence.**

---

## 3. Model Architectures & SHAP Specifications

### 3.1 Task 2 Model: Random Forest Classifier
- **Algorithm**: `sklearn.ensemble.RandomForestClassifier`
- **Hyperparameters**: `n_estimators=150`, `max_depth=10`, `random_state=42`, `n_jobs=-1`
- **Explainer Class**: `shap.TreeExplainer`
- **Output Scale**: Marginal probability (`model_output='probability'`, `class_index=1`)
- **Background**: 100 randomly sampled training instances (`random_state=42`)
- **Mathematical Additivity Check**:
  - Base Value ($\phi_0$): `0.227249`
  - Reconstructed output for Test Sample 0: `0.227249 + 0.025829 = 0.253078`
  - Model `predict_proba`: `0.253078`
  - Reconstruction residual: `1.1706e-09` (verified within machine floating-point precision)

### 3.2 Task 6 Model: RBF Support Vector Classifier (Alternative Black Box)
- **Algorithm**: `sklearn.pipeline.Pipeline([('scaler', StandardScaler()), ('svm', SVC(kernel='rbf', C=1.0, probability=True, random_state=42))])`
- **Explainer Class**: `shap.KernelExplainer` (model-agnostic weighted linear regression)
- **Output Scale**: Marginal probability function `svm_pipe.predict_proba(x)[:, 1]`
- **Background**: 50 representative training medoids derived via `shap.kmeans(X_train, 50)`
- **Computation Settings**: `nsamples=500` for local explanations; `nsamples=300` for global sampling
- **Mathematical Additivity Check**:
  - Base Value ($\phi_0$): `0.302458`
  - Reconstructed output for Case #159: `0.302458 + 0.543482 = 0.845940`
  - Model `predict_proba`: `0.845940`
  - Reconstruction residual: `0.000000e+00` (exact closed-form least squares convergence)

---

## 4. Head-to-Head Performance & Metric Summary

Evaluated on the 1,022 held-out test instances at decision threshold $T = 0.50$:

| Metric | Task 2 Model (Random Forest) | Task 6 Model (RBF Support Vector Machine) | Clinical Interpretation |
| :--- | :---: | :---: | :--- |
| **Accuracy** | **0.8875** | 0.7750 | Driven by majority negative class (95.1%) |
| **Balanced Accuracy** | 0.6468 | **0.6825** | Unweighted mean of specificity and sensitivity |
| **Precision (Class 1)** | **0.1845** | 0.1218 | Constrained by natural low prevalence (4.89%) |
| **Recall (Class 1)** | 0.3800 | **0.5800** | **SVM caught 29 of 50 stroke cases (RF caught 19)** |
| **F1-Score** | **0.2484** | 0.2014 | Harmonic mean of precision and recall |
| **ROC AUC** | **0.7915** | 0.7758 | Overall ranking discrimination on probabilities |
| **Average Precision (PR AUC)** | 0.1694 | **0.1723** | ~3.5× higher than natural random baseline (0.0489) |
| **Confusion Matrix** | TN: 888, FP: 84<br>FN: 31, TP: 19 | TN: 763, FP: 209<br>FN: 21, TP: 29 | **SVM reduces missed stroke cases (FN: 21 vs 31)** |

---

## 5. Artifact Directory & Submission Structure

The final deliverables and all supporting evidence files are structured as follows:

### Primary Deliverables (Project Root)
- **`Stroke_SHAP_Submission_Report.docx`**: Fully compiled and audited final submission report with 100% of instruction text and template bracket placeholders replaced, and all 5 high-resolution figures embedded.
- **`Task5_Stroke_SHAP_Three_Audience_Slides.pptx`**: Standalone presentation slide deck formatted for Data Scientist, Doctor, and Patient audiences.
- **`README.md`**: Technical reproducibility appendix and study documentation.

### Supporting Evidence & Output Tree (`output/`)
```
output/
├── task1_eda/
│   └── figure1_eda_panel.png                   # 6-panel comprehensive exploratory analysis figure
├── task2_shap/
│   ├── metrics.json                            # Serialized evaluation metrics for Random Forest
│   ├── X_train_enc.csv                         # Fully encoded, preprocessed training split
│   ├── X_test_enc.csv                          # Untouched held-out test split
│   ├── y_train.csv / y_test.csv                # Target split series
│   └── y_prob.npy                              # Predicted probability array for test instances
├── task3_force_plot/
│   ├── figure2_local_force_plot.png            # Static publication-quality Force Plot for Case #159
│   ├── figure2_local_force_plot.html           # Interactive HTML Force Plot
│   └── case_summary.json                       # Case #159 clinical values and SHAP breakdown
├── task4_summary_plot/
│   ├── figure3_global_summary_plot.png         # Beeswarm summary plot (N=1,022 test cases)
│   ├── global_feature_importance_bar.png       # Mean |SHAP| feature ranking bar chart
│   ├── global_feature_ranking.csv              # Full quantitative ranking table
│   └── shap_vals_test_c1.npy                   # Serialized SHAP values for test split
├── task5_slides/
│   ├── Task5_Stroke_SHAP_Three_Audience_Slides.pptx  # Slide deck copy
│   └── patient_friendly_factors.png            # Simplified horizontal bar visual for Slide 3
└── task6_alternative_model/
    ├── metrics_svm.json                        # Serialized evaluation metrics for SVM
    ├── figure4_local_svm_force_plot.png        # Local Kernel SHAP force plot for Case #159
    ├── figure4_local_svm_force_plot.html       # Interactive HTML Kernel SHAP plot
    ├── figure5_global_svm_summary_plot.png     # Global Kernel SHAP beeswarm plot (N=150)
    ├── svm_feature_ranking.csv                 # Feature ranking table for SVM
    └── comparison_summary.json                 # Comparative audit metrics
```

---

## 6. How to Reproduce All Results

To replicate the complete pipeline from scratch using the saved environment:

```bash
# 1. Activate environment
conda activate xai_env

# 2. Verify dataset path
# Ensure healthcare-dataset-stroke-data.csv is inside xai-series-master/xai-series-master/data/

# 3. All outputs were compiled using exact seeds (seed=42)
# All preprocessed arrays and metrics match the published report tables exactly.
```

