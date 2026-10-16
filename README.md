# ML Capstone — 23CSE301 (AY 2026-27)

End-to-end Machine Learning pipeline covering three tracks — **Regression**, **Classification** and **Clustering** — for the 23CSE301 Machine Learning capstone project (B.Tech. CSE, III Year).

## Project overview

| | |
|---|---|
| Course | 23CSE301 Machine Learning |
| Academic Year | 2026-27 |
| Team size | 3 |
| Tracks | Regression, Classification, Clustering |
| Tools | Python 3.12, scikit-learn, pandas, NumPy, Matplotlib, Seaborn |

### Team details

| Roll No | Member |
|---|---|
| CB.SC.U4CSE24319 | Girithar R |
| CB.SC.U4CSE24327 | Ravi Teja KV |
| CB.SC.U4CSE24351 | Rushil |

## Problem statement

Regression and Classification use **separate datasets**, each chosen to fit its problem naturally.

| Track | Dataset | Target | Framing |
|---|---|---|---|
| Regression | [Used Cars](https://www.kaggle.com/datasets/thedevastator/uncovering-factors-that-affect-used-car-prices) — `data/raw/autos.csv` (~371k listings) | `price` (continuous) | Predict a used car's price from its listing attributes (vehicle type, age, power, mileage, brand …) |
| Classification | [Vehicle Loan Default Prediction](https://www.kaggle.com/datasets/mamtadhaker/lt-vehicle-loan-default-prediction) — `data/raw/train.csv` (~233k loans) | `loan_default` (binary; ~78 % / 22 %) | Predict whether a vehicle loan defaults from borrower and credit-bureau attributes |
| Clustering | [Indian Vehicle Registration Data (2020–25)](https://www.kaggle.com/datasets/aatifahmad123/indian-vehicle-registration-data-202025) — `data/raw/vehicle_registration/vehicle_registrations.csv` (500k records) | — (unsupervised) | Segment vehicle-registration records by vehicle characteristics (weight, fuel, emission norm, class, sale type, group size); the official vehicle category is held out and used only post-hoc |

Full data dictionaries and cleaning rules: [`data/README.md`](data/README.md).

## Repository structure

```
/
├── README.md               # This file
├── requirements.txt        # Pinned Python dependencies
├── data/
│   ├── raw/                # autos.csv (regression), train.csv (classification), vehicle_registration/vehicle_registrations.csv (clustering)
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
| Regression | Linear, Ridge, Lasso, ElasticNet, Polynomial, Decision Tree, Random Forest, Gradient Boosting, SVR, KNN | Implemented, executed |
| Classification | Logistic Regression, KNN, Gaussian Naive Bayes, Decision Tree, SVC, Random Forest, AdaBoost, Gradient Boosting, Bagging (Decision Tree base), MLP | Implemented, executed — plus consolidated 10-model table, ROC curves, final-model tuning, 5-fold CV and model export |
| Clustering | K-Means (Elbow), Agglomerative (Dendrogram) | Implemented, executed — elbow, dendrogram, linkage comparison, 3 metrics, PCA, t-SNE, cluster profiling |

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

### Classification — all 10 algorithms (`loan_default`, ranked by ROC-AUC)

| Model | Accuracy | F1 (weighted) | ROC-AUC |
|---|---|---|---|
| Random Forest Classifier | 0.6344 | 0.6544 | 0.5790 |
| Logistic Regression | 0.5513 | 0.5902 | 0.5787 |
| MLP Classifier | 0.7819 | 0.6873 | 0.5778 |
| Gaussian Naive Bayes | 0.2763 | 0.2091 | 0.5730 |
| Gradient Boosting Classifier | 0.5625 | 0.5999 | 0.5726 |
| AdaBoost Classifier | 0.5406 | 0.5805 | 0.5721 |
| Bagging Classifier | 0.6113 | 0.6395 | 0.5712 |
| SVC | 0.7813 | 0.6870 | 0.5710 |
| Decision Tree Classifier | 0.5013 | 0.5420 | 0.5600 |
| KNN | 0.7825 | 0.6997 | 0.5548 |

**Final model:** the best model by test ROC-AUC (Random Forest) was tuned with `GridSearchCV` (best: `n_estimators=100, max_depth=6, min_samples_leaf=10`). Test ROC-AUC rose from 0.5790 to 0.5864 (+0.0074), while accuracy and weighted F1 fell (0.6344 → 0.5744, 0.6544 → 0.6110) because the tuned forest is shallower. 5-fold CV ROC-AUC of the two best models: Random Forest 0.618 ± 0.013, Logistic Regression 0.630 ± 0.014. The tuned model is saved as `models/best_classifier.joblib` and used by the GUI.

All ten models separate defaulters only weakly (ROC-AUC 0.55–0.59); the 78 % majority class makes accuracy misleading (MLP, SVC and KNN score ~0.78 by predicting almost everyone as non-default).

### Clustering — Indian vehicle registrations (4,000-record sample, k chosen by silhouette)

| Algorithm | Clusters | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ | ARI vs official category | NMI |
|---|---|---|---|---|---|---|
| K-Means | 2 | 0.2717 | 1.5350 | 1447.45 | 0.6576 | 0.5987 |
| Agglomerative (Ward) | 2 | 0.2530 | 1.6411 | 1315.05 | 0.5954 | 0.5063 |

Both algorithms split the records into a light, high-volume, petrol group (mostly two-wheelers; mean weight ≈ 300–400 kg) and a heavier diesel / motor-car group (≈ 3,000 kg). The official category was never used for fitting; ARI / NMI are post-hoc agreement only. Full output (elbow, dendrogram, linkage comparison, PCA, t-SNE, cluster profiles) is in `notebooks/clustering.ipynb`.

## Project checklist — everything the guidelines require

✅ = done and executed in the notebooks · ✍️ = to be written by the team (see the list below).

### Regression (`notebooks/regression.ipynb`)

| Section | Requirement | Status |
|---|---|---|
| A1 | Dataset audit: shape, dtypes, missing values, target distribution | ✅ |
| A2 | EDA: distribution plots for every feature, correlation heatmap, target distribution, ≥ 2 feature–target scatter plots | ✅ |
| A3 | Written observation next to each major plot | ✅ original plots · ✍️ added plots |
| B1–B3 | Cleaning (missing values, duplicates, outliers), encoding / scaling fitted on train only / stratified split, engineered feature | ✅ |
| C1 | All 10 algorithms, each with its note: coefficients (Linear), `alpha` (Ridge), sparsity (Lasso), `l1_ratio` (ElasticNet), degrees (Polynomial), `max_depth` + importance (Decision Tree), `n_estimators` (Random Forest), `learning_rate` (Gradient Boosting), `C` + kernel (SVR), `k` + scaling (KNN) | ✅ · ✍️ interpretation |
| C2 | R² / RMSE / MAE table ranked by R²; 5-fold CV R² of the top 2 | ✅ |
| C3 | `GridSearchCV` on ≥ 2 models | ✅ |
| C4 | Residual plot, predicted-vs-actual, feature importance | ✅ |

### Classification (`notebooks/classification.ipynb`)

| Section | Requirement | Status |
|---|---|---|
| A1–A3 | Audit, EDA (all feature distributions, heatmap, class balance, scatter plots), observations | ✅ · ✍️ added plots |
| B1–B3 | Cleaning (duplicates / outliers checked, missing values), encoding, scaling, **stratified** split, engineered features | ✅ |
| C1–C3 | Algorithms 1–10, each with its note: odds ratios (Logistic Regression), `k` + distance metrics (KNN), independence assumption (Gaussian NB), `max_depth` + tree plot (Decision Tree), `C` + kernel (SVC), importance + `n_estimators` (Random Forest), tuning (AdaBoost, MLP), Decision Tree base (Bagging) | ✅ · ✍️ interpretation |
| C2 / D1 | Accuracy, Precision, Recall, F1, confusion matrix per algorithm; one consolidated 10-algorithm table | ✅ |
| D2 | ROC curves for all 10 | ✅ |
| D3 | Best model justified, tuned, improvement documented; 5-fold CV of the top 2; model exported | ✅ numbers · ✍️ justification |

### Clustering (`notebooks/clustering.ipynb`)

| Section | Requirement | Status |
|---|---|---|
| A–B | Dataset audit, EDA, preprocessing (labels held out) | ✅ · ✍️ observation, justification |
| C1 / C2 | K-Means with elbow curve; Agglomerative with dendrogram and linkage comparison | ✅ · ✍️ observations |
| D | Silhouette, Davies-Bouldin, Calinski-Harabasz for both algorithms | ✅ · ✍️ interpretation |
| E | PCA 2-D plot for every algorithm; t-SNE | ✅ · ✍️ observation |
| F | Post-hoc validation against the held-out category, cluster profiles | ✅ · ✍️ interpretation |

### Project-level deliverables

| Deliverable | Status |
|---|---|
| Notebooks clean, fully run, outputs visible | ✅ |
| GitHub repository with commit history, `requirements.txt` | ✅ |
| README (datasets, problem statement, results, environment, how to run, AI disclosure) | ✅ |
| Repository structure (`data/`, `notebooks/`, `models/`, `app/`) | ✅ |
| Bonus: Streamlit GUI | ✅ works locally |
| Bonus: public deployment | ✍️ needs your Streamlit / Hugging Face account — steps in `app/README.md`, then paste the URL there |
| Presentation / viva | ✍️ team |

### Cells the team still has to write (analysis and interpretation, per guidelines 7.5)

| Notebook | Where | Cells |
|---|---|---|
| `regression.ipynb` | A2 (categorical plots); C1 supplementary analyses 1b, 3b, 4b, 5b, 6b, 7b, 8b, 9b, 10b; Conclusion | 11 |
| `classification.ipynb` | A2 (3 added plots); C1 analyses 1b, 2b, 3b; D1 consolidated table; D3 final-model justification; Conclusion | 9 |
| `clustering.ipynb` | A2; B preprocessing justification; C1, C2, D, E, F; Conclusion | 8 |

Each is a markdown cell titled "(write your own)" or "TODO (write your own)".

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
3. All three notebooks run end-to-end with outputs saved. Only the **"write your own" observation / justification / conclusion cells are left blank on purpose** — the course guidelines require analysis and interpretation to be the team's own work.
4. Bonus GUI: see [`app/README.md`](app/README.md).

## AI assistance disclosure

_Required by the course guidelines (7.5). **Team: check this paragraph against what you actually did and correct it if needed.**_

Generative AI (Claude, by Anthropic) was used for **code**: the repository and notebook framework; data loading, cleaning, EDA and preprocessing code; the algorithm code for classification algorithms 6–10 and clustering (classification algorithms 6–10, K-Means and Agglomerative clustering) together with their tuning, evaluation and plotting code; the model-export code and the Streamlit GUI; and first-draft wording of some Markdown observations. The regression algorithm cells and classification algorithms 1–5 were written / edited by team members (see commit history); classification algorithms 6–10, the clustering algorithms and the supplementary analysis cells (coefficients, sparsity, degree / depth / learning-rate / k sweeps, distance metrics, extra EDA plots) were written with Claude. **The observation, justification and conclusion cells left as "write your own" are the team's own analysis.** External code, where used, is cited in Markdown cells.

## Academic integrity

Code from external sources (StackOverflow, blogs …) is cited in a Markdown cell next to where it is used.
