# app/ — AutoML Studio Web Dashboard

Minimalist, high-contrast monochrome (pure black & white) web application covering all tracks of the 23CSE301 ML Capstone:
- **Track 1 (Classification):** Vehicle Loan Default Predictor (evaluates borrower risk under 78/22 imbalance)
- **Track 2 (Clustering):** Automotive Fleet Segmentation (monochrome 2D PCA cluster space, K-Means vs Ward, cluster profiles)
- **Track 3 (Regression):** Vehicle Resale Price Estimator (predicts market valuation from specs)
- **Track 4 (Benchmarks):** Review 2 Rubric & 10-Algorithm Comparison Benchmark

| File | Purpose |
|---|---|
| `server.py` | Flask REST API & Web Application server (defaults to port 5050) |
| `templates/index.html` | Clean, user-friendly black & white dashboard with theme toggle button |
| `static/css/style.css` | Strictly monochrome design system supporting Dark Mode and Light Mode |
| `static/js/app.js` | Frontend logic, real-time prediction bindings, theme toggle persistence, 2D PCA canvas |
| `app.py` | Multi-track Streamlit GUI interface |
| `requirements.txt` | Deploy-time dependencies (Flask, scikit-learn, joblib, pandas, numpy) |

## Models Used

The app loads pre-trained artifacts from the `models/` directory:
1. `models/best_classifier.joblib`: Tuned Random Forest Classifier for loan default risk
2. `models/best_regressor.joblib`: Random Forest Regressor for used car price estimation
3. `models/clustering_artifacts.joblib`: Precomputed 2D PCA coordinates, K-Means & Agglomerative models, cluster profiles

## Run Locally (Flask Unified Dashboard)

```bash
# 1. Install dependencies
pip install -r app/requirements.txt

# 2. Start the web application
python app/server.py
```
Open **[http://127.0.0.1:5050](http://127.0.0.1:5050)** in your browser.

## Alternative: Run Legacy Streamlit App

```bash
streamlit run app/app.py
```
