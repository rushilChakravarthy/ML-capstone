# models/

Saved model artifacts (via `joblib.dump`). `*.pkl` / `*.joblib` are git-ignored **except** `best_classifier.joblib`, which the GUI in `app/` needs for deployment.

| File | Track | Algorithm | Notes |
|---|---|---|---|
| `best_classifier.joblib` | Classification | Tuned Random Forest (`n_estimators=100, max_depth=6, min_samples_leaf=10`) | Fitted `Pipeline([("pre", preprocessor), ("clf", best_model)])`; loaded by `app/app.py` |
