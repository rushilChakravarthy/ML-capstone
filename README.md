# ML Capstone — 23CSE301 (AY 2026-27)

End-to-end Machine Learning pipeline covering three tracks — **Regression**, **Classification** and **Clustering** — for the 23CSE301 Machine Learning capstone project (B.Tech. CSE, III Year).

## Project overview

| | |
|---|---|
| Course | 23CSE301 Machine Learning |
| Academic Year | 2026-27 |
| Team size | 3 |
| Tracks | Regression, Classification, Clustering |
| Reviews | Review 1: Regression + Classification Part A · Review 2: Classification Part B + Clustering |
| Tools | Python 3.12, scikit-learn, pandas, NumPy, Matplotlib, Seaborn |

### Team & track ownership

| Member | Primary track (owner) |
|---|---|
| _TBD_ | _TBD_ |
| _TBD_ | _TBD_ |
| _TBD_ | _TBD_ |

## Problem statement

Regression and Classification use **separate datasets**, each chosen to fit its problem naturally.

| Track | Dataset | Target | Framing |
|---|---|---|---|
| Regression | [Used Cars](https://www.kaggle.com/datasets/thedevastator/uncovering-factors-that-affect-used-car-prices) — `data/raw/autos.csv` (~371k listings) | `price` (continuous) | Predict a used car's price from its listing attributes (vehicle type, age, power, mileage, brand …) |
| Classification | [Vehicle Loan Default Prediction](https://www.kaggle.com/datasets/mamtadhaker/lt-vehicle-loan-default-prediction) — `data/raw/train.csv` (~233k loans) | `loan_default` (binary; ~78 % / 22 %) | Predict whether a vehicle loan defaults from borrower and credit-bureau attributes |
| Clustering | [Indian Vehicle Registration Data (2020–25)](https://www.kaggle.com/datasets/aatifahmad123/indian-vehicle-registration-data-202025) — fetched into `data/raw/vehicle_registration/` | — (unsupervised) | Group vehicle-registration records by their attributes; ground truth (if any) used only post-hoc |

Full data dictionaries and cleaning rules: [`data/README.md`](data/README.md).

## Repository structure

```
/
├── README.md               # This file
├── requirements.txt        # Pinned Python dependencies
├── data/
│   ├── raw/                # autos.csv (regression), train.csv (classification), vehicle_registration/ (clustering, downloaded)
│   ├── download_dataset.py # Kaggle fetch script for autos.csv (fallback)
│   └── README.md           # Dataset descriptions + data dictionaries
├── notebooks/
│   ├── regression.ipynb
│   ├── classification.ipynb
│   └── clustering.ipynb
├── models/                 # Saved models (best_classifier.joblib is used by the GUI)
└── app/                    # Streamlit GUI for the bonus marks
```

## Tracks, algorithms and status

| Track | Algorithms | Status |
|---|---|---|
| Regression (Review 1) | Linear, Ridge, Lasso, ElasticNet, Polynomial, Decision Tree, Random Forest, Gradient Boosting, SVR, KNN | Implemented, executed |
| Classification Part A (Review 1) | Logistic Regression, KNN, Gaussian Naive Bayes, Decision Tree, SVC | Implemented, executed |
| Classification Part B (Review 2) | Random Forest, AdaBoost, Gradient Boosting, Bagging (Decision Tree base), MLP | **Only the 5 algorithm cells are `TODO`**; the consolidated table, ROC curves, final-model tuning, 5-fold CV and model saving for the GUI are pre-built and run automatically once the 5 models exist |
| Clustering (Review 2) | K-Means (Elbow), Agglomerative (Dendrogram) | **Only the K-Means / Agglomerative cells (elbow, dendrogram, fits) are `TODO`**; EDA, preprocessing, metrics, PCA, t-SNE and cluster profiling are pre-built. Dataset must be downloaded first (`python data/download_dataset.py vehicle-reg`) |

Metrics — Regression: R², RMSE, MAE, 5-fold CV R² (top 2). Classification: Accuracy, Precision, Recall, weighted F1, confusion matrix, ROC-AUC (binary target, so plain ROC-AUC rather than One-vs-Rest). Clustering: Silhouette, Davies-Bouldin, Calinski-Harabasz; PCA 2-D plot for every algorithm, t-SNE for at least one.

## Results summary

Hold-out test set (80/20 split, `random_state=42`) on an 8,000-row random sample of each cleaned dataset (sampled so SVR / SVC / GridSearchCV run in reasonable time).

### Regression — `price` (ranked by R²)

| Model | R² | RMSE | MAE |
|---|---|---|---|
| Random Forest Regressor | 0.8203 | 3191.56 | 1584.00 |
| SVR | 0.8047 | 3327.80 | 1536.36 |
| Gradient Boosting Regressor | 0.8013 | 3356.11 | 1748.07 |
| Polynomial Regression | 0.7626 | 3668.86 | 2146.64 |
| Decision Tree Regressor | 0.7503 | 3762.83 | 1983.95 |
| KNN Regressor | 0.7258 | 3943.00 | 2131.22 |
| Ridge Regression | 0.6852 | 4224.61 | 2620.29 |
| Lasso Regression | 0.6796 | 4261.71 | 2624.88 |
| Linear Regression | 0.6785 | 4269.16 | 2629.97 |
| ElasticNet Regression | 0.6632 | 4369.68 | 2737.10 |

### Classification Part A — `loan_default` (ranked by ROC-AUC)

| Model | Accuracy | F1 (weighted) | ROC-AUC |
|---|---|---|---|
| Logistic Regression | 0.5513 | 0.5902 | 0.5787 |
| Gaussian Naive Bayes | 0.2763 | 0.2091 | 0.5730 |
| SVC | 0.7813 | 0.6870 | 0.5710 |
| Decision Tree Classifier | 0.5013 | 0.5420 | 0.5600 |
| KNN | 0.7825 | 0.6997 | 0.5548 |

Precision / Recall and per-model confusion matrices are in `notebooks/classification.ipynb`. All models separate the classes only weakly (ROC-AUC ≈ 0.55-0.58): the signal is diffuse across many weak features, and the high accuracy of KNN / SVC mostly reflects predicting the 78 % majority class.

_Part B (6-10), the consolidated 10-algorithm table and the clustering results are added in Review 2 — see the notebooks._

## Environment setup

```bash
git clone https://github.com/Rushil-99/ML-capstone.git
cd ML-capstone
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

## How to run

1. Both datasets used so far are already committed in `data/raw/` — no download step. (`python data/download_dataset.py vehicle-reg` fetches the clustering dataset from Kaggle; it needs a free Kaggle API token — see the script's docstring.)
2. Open a notebook in `notebooks/` and **Run All** (top to bottom). `random_state=42` is used throughout.
3. `regression.ipynb` and `classification.ipynb` (Part A) run end-to-end with outputs saved. The only cells left for the team are the algorithm cells marked `# TODO` (classification Part B 6-10; K-Means and Agglomerative in `clustering.ipynb`) and the "write your own" observation cells. Each algorithm cell fills a shared dict (`fitted_models` / `predictions` / `probabilities`, or `cluster_labels`) and everything after it runs automatically. Download the clustering data first: `python data/download_dataset.py vehicle-reg`.
4. Bonus GUI: see [`app/README.md`](app/README.md).

## AI assistance disclosure

_Required by the course guidelines (7.5). **Team: review and edit this paragraph so it reflects exactly what was done.**_

Generative AI (Claude, by Anthropic) was used for **code scaffolding**: the repository/notebook framework, dataset loading, cleaning and feature-engineering code, evaluation / plotting scaffolding, the Streamlit GUI scaffold, and first-draft wording of some Markdown observations. Algorithm implementations, hyperparameter choices, analysis and interpretation are the team's own work. External code, where used, is cited in Markdown cells.

## Academic integrity

Code from external sources (StackOverflow, blogs …) is cited in a Markdown cell next to where it is used.
