# app/ — Streamlit GUI (bonus)

Interactive GUI for the loan-default classifier — worth **+1** for a working GUI and **+1** more for a live public deployment (max +2, Review 2 rubric).

| File | Purpose |
|---|---|
| `app.py` | Streamlit app: form → `predict_proba` → default probability |
| `requirements.txt` | Deploy-time dependencies (pinned) |

## Before it works: save the model
`app.py` loads `models/best_classifier.joblib`. The final-model cell in `classification.ipynb` (Section A3) saves it automatically once all 10 models exist. To save a model manually:

```python
import joblib
from sklearn.pipeline import Pipeline
joblib.dump(Pipeline([("pre", preprocessor), ("clf", best_model)]), "../models/best_classifier.joblib")
```
Without that file the app shows an explanatory message instead of crashing.

## Run locally
```bash
pip install -r app/requirements.txt
streamlit run app/app.py
```

## Deploy (for the second bonus mark)
Commit `models/best_classifier.joblib` (it is explicitly un-ignored in `.gitignore`), then deploy `app/app.py` on Streamlit Community Cloud / Hugging Face Spaces / Render and paste the public URL here:

**Public URL:** _TBD_
