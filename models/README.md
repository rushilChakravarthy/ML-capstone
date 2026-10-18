# models/

Saved model artifacts (via `joblib.dump`) used by the Capstone UI.

| File | Track | Algorithm | Notes |
|---|---|---|---|
| `best_classifier.joblib` | Classification | Tuned Random Forest (`n_estimators=100, max_depth=6, min_samples_leaf=10`) | Fitted `Pipeline([("pre", preprocessor), ("clf", best_model)])`; ROC-AUC = 0.5864; loaded by the web UI |
| `best_regressor.joblib` | Regression | Random Forest Regressor (`n_estimators=100`) | Fitted `Pipeline([("pre", preprocessor), ("reg", reg_model)])`; Test R² = 0.8203, MAE = €1,584; loaded by the web UI |
| `clustering_artifacts.joblib` | Clustering | K-Means (k=2) & Agglomerative (Ward) | Fitted scalers, OneHotEncoder, PCA projections (2D), cluster profiles, and 4,000-point sample coordinates |
