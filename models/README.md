# models/

Saved model artifacts (via `joblib.dump`) — optional per the guidelines. `*.pkl` / `*.joblib` are git-ignored **except** `best_classifier.joblib`, which the GUI in `app/` needs for deployment.

| File | Track | Algorithm | Notes |
|---|---|---|---|
| `best_classifier.joblib` | Classification | _TBD — final model from Part B_ | Fitted `Pipeline([("pre", preprocessor), ("clf", best_model)])`; loaded by `app/app.py` |
