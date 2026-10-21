# ML Capstone — 23CSE301 (AY 2026-27)

End-to-end Machine Learning pipeline covering three tracks — **Regression**, **Classification** and **Clustering** — for the 23CSE301 Machine Learning capstone project .

## Project overview

| | |
|---|---|
| Course | 23CSE301 Machine Learning |
| Tracks | Regression, Classification, Clustering |
| Tools | Python 3.12, scikit-learn, pandas, NumPy, Matplotlib, Seaborn |

### Team details

| Roll No | Member |
|---|---|
| CB.SC.U4CSE24319 | Girithar R |
| CB.SC.U4CSE24327 | Ravi Teja K V |
| CB.SC.U4CSE24351 | Rushil T |

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
├── models/                 # Saved models (best_classifier.joblib, best_regressor.joblib, clustering_artifacts.joblib)
└── app/                    # Unified Web Dashboard (Flask + HTML/CSS/JS) covering all 3 tracks
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

Single consolidated table comparing all 10 classification algorithms on Accuracy, Precision, Recall, F1, and ROC-AUC (Rubric A2):

| Rank | Model | Accuracy | Precision | Recall | F1 (weighted) | ROC-AUC |
|---|---|---|---|---|---|---|
| 1 | **Random Forest Classifier** | 0.6344 | 0.6827 | 0.6344 | 0.6544 | **0.5790** |
| 2 | **Logistic Regression** | 0.5513 | 0.7006 | 0.5513 | 0.5902 | **0.5787** |
| 3 | **MLP Classifier** | 0.7819 | 0.7207 | 0.7819 | 0.6873 | **0.5778** |
| 4 | **Gaussian Naive Bayes** | 0.2763 | 0.7012 | 0.2763 | 0.2091 | **0.5730** |
| 5 | **Gradient Boosting Classifier** | 0.5625 | 0.6829 | 0.5625 | 0.5999 | **0.5726** |
| 6 | **AdaBoost Classifier** | 0.5406 | 0.6928 | 0.5406 | 0.5805 | **0.5721** |
| 7 | **Bagging Classifier (DT base)** | 0.6113 | 0.6881 | 0.6113 | 0.6395 | **0.5712** |
| 8 | **SVC (Linear Calibrated)** | 0.7813 | 0.6842 | 0.7813 | 0.6870 | **0.5710** |
| 9 | **Decision Tree Classifier** | 0.5013 | 0.6883 | 0.5013 | 0.5420 | **0.5600** |
| 10 | **KNN Classifier** | 0.7825 | 0.7286 | 0.7825 | 0.6997 | **0.5548** |

**Final model selection & hyperparameter tuning (Rubric A3):**
The best model by test ROC-AUC (Random Forest) was tuned with `GridSearchCV` (optimal: `n_estimators=100, max_depth=6, min_samples_leaf=10`). Test ROC-AUC rose from 0.5790 to **0.5864 (+0.0074 / +0.74%)**, regularizing tree depth to prevent overfitting. 5-fold CV ROC-AUC on the training set confirmed stable discrimination at **0.6179 ± 0.0125** for Random Forest (and 0.6300 ± 0.0136 for Logistic Regression). The tuned model pipeline is exported to `models/best_classifier.joblib` and powers the real-time prediction GUI.

*Class imbalance note:* Across the 78/22 class imbalance, Accuracy alone is deceptive (models predicting the majority class score ~78% but have near-zero recall on defaulters). ROC-AUC provides the true threshold-invariant discrimination metric.

### Clustering — Indian vehicle registrations (4,000-record sample, k chosen by silhouette)

| Algorithm | Clusters | Silhouette ↑ | Davies-Bouldin ↓ | Calinski-Harabasz ↑ | ARI vs official category | NMI |
|---|---|---|---|---|---|---|
| **K-Means** | 2 | **0.2717** | **1.6360** | **1205.85** | **0.6576** | **0.5987** |
| **Agglomerative (Ward)** | 2 | **0.2530** | **1.7088** | **1047.88** | **0.5954** | **0.5063** |

Both algorithms split the records into a light, high-volume, petrol group (mostly two-wheelers; mean weight ≈ 300–400 kg) and a heavier diesel / commercial group (≈ 3,000 kg). The official category was never used for fitting; ARI / NMI are post-hoc agreement only. Full output (elbow, dendrogram, linkage comparison, PCA, t-SNE, cluster profiles) is in `notebooks/clustering.ipynb`.

## Review 2 Rubric Compliance (25 Marks + Up to 2 Bonus)

| Section | Criterion | Requirement | Project Implementation |
|---|---|---|---|
| **A1** | Implementation (3M) | All 5 Part-B algorithms trained on same dataset as Part A, no errors | RF, AdaBoost, Gradient Boosting, Bagging, MLP implemented and evaluated on identical 80/20 split. |
| **A2** | Comparison Table (1M) | Single table comparing all 10 algorithms on Acc, Prec, Rec, F1, ROC-AUC | Complete 10-algorithm table rendered above, in `classification.ipynb`, and in GUI. |
| **A3** | Final Model Selection (2M) | Best model identified, tuned via GridSearchCV, metric improvement documented | Random Forest selected (Top ROC-AUC 0.5790); Tuned (`max_depth:6, min_samples_leaf:10`) boosting ROC-AUC to **0.5864 (+0.74%)**. |
| **B1** | Clustering Implementation (4M) | All 2 required clustering algorithms fitted, cluster labels obtained | K-Means ($k=2$) and Agglomerative ($k=2$, Ward) fitted on scaled RTO fleet data without errors. |
| **B2** | Evaluation Metrics (2M) | Silhouette, Davies-Bouldin, Calinski-Harabasz reported; Elbow & Dendrogram shown | All 3 metrics reported; Elbow & Silhouette curves plotted for $k=2..8$; Dendrogram plotted for Hierarchical. |
| **B3** | Cluster Visualisation (2M) | PCA-reduced 2D scatter for every algorithm; t-SNE plot for at least one | PCA 2D scatter plots for both algorithms; t-SNE 2D manifold plots for both algorithms. |
| **C1** | Code Quality (1M) | Clean top-to-bottom notebooks, modular functions, documented steps | All notebooks run end-to-end without errors; all observations and conclusions rigorously completed. |
| **C2** | README & Documentation (1M) | Dataset description, problem statement, results summary, environment setup | Complete problem statements, data dictionaries, full metrics table, and run instructions provided. |
| **C3** | GitHub Repository (1M) | Meaningful commit history; `requirements.txt` present | Clean multi-stage commit history, `requirements.txt`, and reproducible scripts. |
| **D1-D2**| Presentation & Viva (8M) | Problem → data → method → results → conclusion arc; comprehensive viva prep | Complete slide deck and viva question bank available in [`docs/REVIEW_2_VIVA_GUIDE.md`](docs/REVIEW_2_VIVA_GUIDE.md). |
| **Bonus**| Interactive GUI & Deployment (+2M) | Working web GUI (Streamlit/Gradio) + Live deployment readiness | Both **Streamlit GUI** (`app/app.py`) and **Flask Dashboard** (`app/server.py`) with 1-click cloud deployment. |

## Environment setup

```bash
git clone https://github.com/Rushil-99/ML-capstone.git
cd ML-capstone
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

## How to run

1. **Notebooks:** Open any notebook in `notebooks/` (`classification.ipynb`, `clustering.ipynb`, `regression.ipynb`) and select **Run All**. All cells run top-to-bottom deterministically (`random_state=42`).
2. **Interactive Streamlit GUI (Bonus +1/+2 Marks):**
   ```bash
   streamlit run app/app.py
   ```
   Opens the interactive multi-track Streamlit dashboard for real-time loan default scoring, 2D PCA/t-SNE fleet cluster exploration, and resale price estimation.
3. **Full-Featured Flask Web Dashboard:**
   ```bash
   python3 app/server.py
   ```
   Opens the dark-mode web dashboard at `http://localhost:5050` (or `http://127.0.0.1:5050`) featuring live predictions, interactive scatter plots, and REST API endpoints.

## AI assistance disclosure

Generative AI (Claude, by Anthropic) was used solely to assist with the exploratory data analysis (EDA) and data preprocessing code. All other components—including the repository and notebook framework, data loading, algorithm implementations, model tuning, evaluation, plotting, supplementary analyses, the Streamlit app, and all observation, justification, and conclusion cells—were developed or written directly by the team. External code, where used, is cited in Markdown cells.
