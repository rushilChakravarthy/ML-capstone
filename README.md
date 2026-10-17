# ML Capstone — 23CSE301 (AY 2026-27)

End-to-end Machine Learning pipeline covering three tracks — **Regression**, **Classification** and **Clustering** — for the 23CSE301 Machine Learning capstone project .

## Project overview

| | |
|---|---|
| Course | 23CSE301 Machine Learning |
| Academic Year | 2026-27 |
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
└── app/                    # Streamlit app for the loan-default classifier
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

### Classification — (`loan_default`, ranked by ROC-AUC)

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
3. All three notebooks run end-to-end with outputs saved. The observation, justification and conclusion cells marked "write your own" are left for the team to write.
4. Streamlit app: see [`app/README.md`](app/README.md).

## AI assistance disclosure

_**Team: check this paragraph against what you actually did and correct it if needed.**_

Generative AI (Claude, by Anthropic) was used for **code**: the repository and notebook framework; data loading, cleaning, EDA and preprocessing code; the algorithm code for classification algorithms 6–10 and for K-Means and Agglomerative clustering, with their tuning, evaluation and plotting code; the supplementary analysis cells (coefficients, sparsity, degree / depth / learning-rate / k sweeps, distance metrics, extra EDA plots); the model-export code and the Streamlit app; and first-draft wording of some Markdown observations. The regression algorithm cells and classification algorithms 1–5 were written / edited by team members (see commit history). **The observation, justification and conclusion cells marked "write your own" are the team's own analysis.** External code, where used, is cited in Markdown cells.
