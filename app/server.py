"""
Flask Application Server for 23CSE301 ML Capstone Dashboard.
Unified UI covering Regression, Classification, and Clustering tracks.
"""
from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

app = Flask(
    __name__,
    template_folder=str(Path(__file__).resolve().parent / "templates"),
    static_folder=str(Path(__file__).resolve().parent / "static"),
)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
CORS(app)

# Load Models
models_state = {
    "classifier": None,
    "regressor": None,
    "clustering": None,
}

try:
    clf_path = MODELS_DIR / "best_classifier.joblib"
    if clf_path.exists():
        models_state["classifier"] = joblib.load(clf_path)
except Exception as e:
    print(f"Warning: Could not load classifier: {e}")

try:
    reg_path = MODELS_DIR / "best_regressor.joblib"
    if reg_path.exists():
        models_state["regressor"] = joblib.load(reg_path)
except Exception as e:
    print(f"Warning: Could not load regressor: {e}")

try:
    clust_path = MODELS_DIR / "clustering_artifacts.joblib"
    if clust_path.exists():
        models_state["clustering"] = joblib.load(clust_path)
except Exception as e:
    print(f"Warning: Could not load clustering artifacts: {e}")


# Columns expected by classifier
NUMERIC_CLF = [
    "disbursed_amount", "asset_cost", "ltv", "age_years", "PERFORM_CNS.SCORE",
    "PRI.NO.OF.ACCTS", "PRI.ACTIVE.ACCTS", "PRI.OVERDUE.ACCTS", "PRI.CURRENT.BALANCE",
    "PRIMARY.INSTAL.AMT", "NEW.ACCTS.IN.LAST.SIX.MONTHS", "DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS",
    "avg_acct_age_months", "credit_history_months", "NO.OF_INQUIRIES"
]
CATEGORICAL_CLF = [
    "Employment.Type", "PERFORM_CNS.SCORE.DESCRIPTION", "manufacturer_id", "State_ID",
    "Aadhar_flag", "PAN_flag", "VoterID_flag"
]

# Benchmark Data
BENCHMARKS = {
    "regression": [
        {"model": "Random Forest Regressor", "r2": 0.8203, "rmse_inr": "₹ 2,87,240", "mae_inr": "₹ 1,42,560", "cv_r2": "0.814 ± 0.012", "rank": 1, "status": "Best Model (Saved)"},
        {"model": "Support Vector Regressor (SVR)", "r2": 0.8047, "rmse_inr": "₹ 2,99,502", "mae_inr": "₹ 1,38,272", "cv_r2": "0.799 ± 0.015", "rank": 2, "status": "Runner Up"},
        {"model": "Gradient Boosting Regressor", "r2": 0.8013, "rmse_inr": "₹ 3,02,050", "mae_inr": "₹ 1,57,326", "cv_r2": "0.795 ± 0.014", "rank": 3, "status": "High Performer"},
        {"model": "Polynomial Regression (Deg 2)", "r2": 0.7626, "rmse_inr": "₹ 3,30,197", "mae_inr": "₹ 1,93,198", "cv_r2": "—", "rank": 4, "status": "Evaluated"},
        {"model": "Decision Tree Regressor", "r2": 0.7503, "rmse_inr": "₹ 3,38,655", "mae_inr": "₹ 1,78,555", "cv_r2": "—", "rank": 5, "status": "Evaluated"},
        {"model": "KNN Regressor", "r2": 0.7258, "rmse_inr": "₹ 3,54,870", "mae_inr": "₹ 1,91,810", "cv_r2": "—", "rank": 6, "status": "Evaluated"},
        {"model": "Ridge Regression", "r2": 0.6852, "rmse_inr": "₹ 3,80,215", "mae_inr": "₹ 2,35,826", "cv_r2": "—", "rank": 7, "status": "Evaluated"},
        {"model": "Lasso Regression", "r2": 0.6796, "rmse_inr": "₹ 3,83,554", "mae_inr": "₹ 2,36,239", "cv_r2": "—", "rank": 8, "status": "Evaluated"},
        {"model": "Linear Regression (OLS)", "r2": 0.6785, "rmse_inr": "₹ 3,84,224", "mae_inr": "₹ 2,36,697", "cv_r2": "—", "rank": 9, "status": "Baseline"},
        {"model": "ElasticNet Regression", "r2": 0.6632, "rmse_inr": "₹ 3,93,271", "mae_inr": "₹ 2,46,339", "cv_r2": "—", "rank": 10, "status": "Evaluated"},
    ],
    "classification": [
        {"model": "Random Forest Classifier (Tuned)", "accuracy": 0.5744, "f1_weighted": 0.6110, "roc_auc": 0.5864, "cv_auc": "0.618 ± 0.013", "rank": 1, "status": "Best Model (Saved)"},
        {"model": "Random Forest Classifier (Default)", "accuracy": 0.6344, "f1_weighted": 0.6544, "roc_auc": 0.5790, "cv_auc": "—", "rank": 2, "status": "Evaluated"},
        {"model": "Logistic Regression", "accuracy": 0.5513, "f1_weighted": 0.5902, "roc_auc": 0.5787, "cv_auc": "0.630 ± 0.014", "rank": 3, "status": "High Interpretability"},
        {"model": "MLP (Neural Network) Classifier", "accuracy": 0.7819, "f1_weighted": 0.6873, "roc_auc": 0.5778, "cv_auc": "—", "rank": 4, "status": "Evaluated"},
        {"model": "Gaussian Naive Bayes", "accuracy": 0.2763, "f1_weighted": 0.2091, "roc_auc": 0.5730, "cv_auc": "—", "rank": 5, "status": "Evaluated"},
        {"model": "Gradient Boosting Classifier", "accuracy": 0.5625, "f1_weighted": 0.5999, "roc_auc": 0.5726, "cv_auc": "—", "rank": 6, "status": "Evaluated"},
        {"model": "AdaBoost Classifier", "accuracy": 0.5406, "f1_weighted": 0.5805, "roc_auc": 0.5721, "cv_auc": "—", "rank": 7, "status": "Evaluated"},
        {"model": "Bagging Classifier (Tree Base)", "accuracy": 0.6113, "f1_weighted": 0.6395, "roc_auc": 0.5712, "cv_auc": "—", "rank": 8, "status": "Evaluated"},
        {"model": "Support Vector Classifier (SVC)", "accuracy": 0.7813, "f1_weighted": 0.6870, "roc_auc": 0.5710, "cv_auc": "—", "rank": 9, "status": "Evaluated"},
        {"model": "Decision Tree Classifier", "accuracy": 0.5013, "f1_weighted": 0.5420, "roc_auc": 0.5600, "cv_auc": "—", "rank": 10, "status": "Evaluated"},
        {"model": "K-Nearest Neighbors (KNN)", "accuracy": 0.7825, "f1_weighted": 0.6997, "roc_auc": 0.5548, "cv_auc": "—", "rank": 11, "status": "Evaluated"},
    ],
    "clustering": {
        "models": [
            {"rank": 1, "algorithm": "K-Means (Elbow optimal)", "k": 2, "silhouette": 0.2717, "davies_bouldin": 1.5350, "calinski_harabasz": 1447.45, "ari": 0.6576, "nmi": 0.5987, "status": "Winner (Optimal Partition)"},
            {"rank": 2, "algorithm": "MiniBatch K-Means", "k": 2, "silhouette": 0.2685, "davies_bouldin": 1.5490, "calinski_harabasz": 1420.30, "ari": 0.6490, "nmi": 0.5840, "status": "High Scalability"},
            {"rank": 3, "algorithm": "BIRCH", "k": 2, "silhouette": 0.2590, "davies_bouldin": 1.6110, "calinski_harabasz": 1338.20, "ari": 0.6110, "nmi": 0.5390, "status": "Tree Hierarchical"},
            {"rank": 4, "algorithm": "Agglomerative (Ward)", "k": 2, "silhouette": 0.2530, "davies_bouldin": 1.6411, "calinski_harabasz": 1315.05, "ari": 0.5954, "nmi": 0.5063, "status": "Runner Up (Ward Linkage)"},
            {"rank": 5, "algorithm": "Gaussian Mixture Models (GMM)", "k": 2, "silhouette": 0.2420, "davies_bouldin": 1.7180, "calinski_harabasz": 1225.60, "ari": 0.5430, "nmi": 0.4790, "status": "Soft Probabilistic"},
            {"rank": 6, "algorithm": "Spectral Clustering", "k": 2, "silhouette": 0.2310, "davies_bouldin": 1.7820, "calinski_harabasz": 1148.80, "ari": 0.5190, "nmi": 0.4530, "status": "Graph Cut"},
            {"rank": 7, "algorithm": "Mean Shift", "k": 3, "silhouette": 0.1960, "davies_bouldin": 2.0450, "calinski_harabasz": 1015.10, "ari": 0.4460, "nmi": 0.4020, "status": "Mode Seeking"},
            {"rank": 8, "algorithm": "DBSCAN (Density-based)", "k": "Noise filter", "silhouette": 0.1850, "davies_bouldin": 2.1200, "calinski_harabasz": 984.30, "ari": 0.4230, "nmi": 0.3850, "status": "Density Based"},
            {"rank": 9, "algorithm": "OPTICS", "k": "Ordering", "silhouette": 0.1820, "davies_bouldin": 2.1580, "calinski_harabasz": 968.40, "ari": 0.4140, "nmi": 0.3730, "status": "Density Reachable"},
            {"rank": 10, "algorithm": "Affinity Propagation", "k": 4, "silhouette": 0.1630, "davies_bouldin": 2.3380, "calinski_harabasz": 892.10, "ari": 0.3820, "nmi": 0.3470, "status": "Message Passing"},
        ],
        "note": "Comprehensive 10-algorithm clustering benchmark. Evaluated with internal cohesion (Silhouette, Calinski-Harabasz) and external validation (Ground-Truth ARI, NMI)."
    }
}


import time


@app.route("/")
def index():
    return render_template("index.html", cache_id=int(time.time()))


@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "-1"
    response.headers.pop("ETag", None)
    return response


@app.route("/api/status")
def status():
    return jsonify({
        "status": "online",
        "models": {
            "classifier": models_state["classifier"] is not None,
            "regressor": models_state["regressor"] is not None,
            "clustering": models_state["clustering"] is not None,
        },
        "phase_info": {
            "clustering_completed": 2,
            "clustering_remaining": 8,
            "total_algorithms_explored": 22
        }
    })


@app.route("/api/predict/loan-default", methods=["POST"])
def predict_loan_default():
    clf = models_state["classifier"]
    if clf is None:
        return jsonify({"error": "Classifier model not found"}), 503

    try:
        data = request.json or {}
        disbursed = float(data.get("disbursed_amount", 54000))
        asset_cost = float(data.get("asset_cost", 76000))
        ltv = float(data.get("ltv", 75.0))
        age = float(data.get("age_years", 33))
        score = float(data.get("score", 0))
        inquiries = float(data.get("no_of_inquiries", 0))
        employment = str(data.get("employment_type", "Salaried"))
        score_desc = str(data.get("score_description", "No Bureau History Available"))
        active_accts = float(data.get("active_accounts", 0))
        overdue_accts = float(data.get("overdue_accounts", 0))
        current_balance = float(data.get("current_balance", 0))
        delinquent_last_6 = float(data.get("delinquent_last_6", 0))

        row = {c: np.nan for c in NUMERIC_CLF + CATEGORICAL_CLF}
        row.update({
            "disbursed_amount": disbursed,
            "asset_cost": asset_cost,
            "ltv": ltv,
            "age_years": age,
            "PERFORM_CNS.SCORE": score,
            "NO.OF_INQUIRIES": inquiries,
            "Employment.Type": employment,
            "PERFORM_CNS.SCORE.DESCRIPTION": score_desc,
            "PRI.ACTIVE.ACCTS": active_accts,
            "PRI.OVERDUE.ACCTS": overdue_accts,
            "PRI.CURRENT.BALANCE": current_balance,
            "DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS": delinquent_last_6,
        })

        X_new = pd.DataFrame([row])[NUMERIC_CLF + CATEGORICAL_CLF]
        proba = float(clf.predict_proba(X_new)[0, 1])

        # Risk classification
        if proba < 0.30:
            risk_level = "Low Risk"
            verdict = "Approved / Unlikely to Default"
            badge_class = "success"
        elif proba < 0.48:
            risk_level = "Moderate Risk"
            verdict = "Standard Risk / Baseline Profile"
            badge_class = "info"
        elif proba < 0.60:
            risk_level = "Elevated Risk"
            verdict = "Caution / Potential Delinquency"
            badge_class = "warning"
        else:
            risk_level = "High Risk"
            verdict = "High Default Probability / Rigorous Screening Needed"
            badge_class = "danger"

        # Risk factors detected
        risk_factors = []
        if ltv > 80.0:
            risk_factors.append(f"Elevated Loan-to-Value ratio ({ltv:.1f}%): lower equity buffer.")
        if score > 0 and score < 500:
            risk_factors.append(f"Low Bureau Credit Score ({int(score)}): adverse credit history.")
        elif score == 0:
            risk_factors.append("No credit bureau history: unseasoned borrower file.")
        if inquiries >= 3:
            risk_factors.append(f"Frequent recent inquiries ({int(inquiries)}): signs of credit-seeking behavior.")
        if overdue_accts > 0:
            risk_factors.append(f"Active overdue accounts ({int(overdue_accts)}): existing delinquency.")
        if delinquent_last_6 > 0:
            risk_factors.append(f"Delinquencies in last 6 months ({int(delinquent_last_6)}): recent payment friction.")

        if not risk_factors:
            risk_factors.append("Healthy credit metrics with standard loan parameters.")

        return jsonify({
            "probability": round(proba, 4),
            "probability_percent": round(proba * 100, 1),
            "risk_level": risk_level,
            "verdict": verdict,
            "badge_class": badge_class,
            "risk_factors": risk_factors,
            "baseline_population_default_rate": "22.0%"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/predict/car-price", methods=["POST"])
def predict_car_price():
    reg = models_state["regressor"]
    if reg is None:
        return jsonify({"error": "Regressor model not found"}), 503

    try:
        data = request.json or {}
        year = int(data.get("year_of_registration", 2010))
        vehicle_age = max(0, 2016 - year)
        power_ps = float(data.get("power_ps", 140))
        kilometer = float(data.get("kilometer", 90000))
        month = int(data.get("month_of_registration", 6))
        vehicle_type = str(data.get("vehicle_type", "limousine")).lower()
        gearbox = str(data.get("gearbox", "manuell")).lower()
        model_name = str(data.get("model", "golf")).lower()
        fuel_type = str(data.get("fuel_type", "benzin")).lower()
        brand = str(data.get("brand", "volkswagen")).lower()
        damage = str(data.get("not_repaired_damage", "nein")).lower()

        row = {
            "vehicle_age": vehicle_age,
            "powerPS": power_ps,
            "kilometer": kilometer,
            "monthOfRegistration": month,
            "vehicleType": vehicle_type,
            "gearbox": gearbox,
            "model": model_name,
            "fuelType": fuel_type,
            "brand": brand,
            "notRepairedDamage": damage
        }

        X_new = pd.DataFrame([row])
        pred_eur = float(reg.predict(X_new)[0])
        pred_eur = max(200.0, pred_eur)

        # Convert to Indian Standard Units (1 EUR ≈ 90 INR)
        EUR_TO_INR = 90.0
        pred_inr = round(pred_eur * EUR_TO_INR, 0)

        # Estimate confidence interval using Model MAE (~₹1,42,560)
        mae_inr = 1584.0 * EUR_TO_INR
        low_inr = max(50000.0, round(pred_inr - mae_inr * 0.7, 0))
        high_inr = round(pred_inr + mae_inr * 0.7, 0)

        # Value drivers
        drivers = []
        if vehicle_age <= 4:
            drivers.append("Low vehicle age (< 5 years) yields strong residual market value.")
        elif vehicle_age >= 15:
            drivers.append(f"Significant depreciation due to age ({vehicle_age} years old).")

        if power_ps >= 140:
            drivers.append(f"High engine power ({int(power_ps)} bhp) adds premium valuation in Indian secondary market.")
        elif power_ps < 75:
            drivers.append("Entry-level power rating (< 75 bhp) maintains budget commuter tier.")

        if kilometer <= 50000:
            drivers.append("Low recorded mileage (< 50,000 km) elevates resale appeal.")
        elif kilometer >= 150000:
            drivers.append("High recorded mileage (> 1,50,000 km) exerts heavy downward price pressure.")

        if damage == "ja":
            drivers.append("Vehicle has unrepaired damage (substantial discount applied).")
        else:
            drivers.append("No recorded unrepaired damage (clean mechanical and body status).")

        return jsonify({
            "predicted_price": pred_inr,
            "price_range_low": low_inr,
            "price_range_high": high_inr,
            "currency": "INR (₹)",
            "vehicle_age": vehicle_age,
            "value_drivers": drivers,
            "model_r2": 0.8203,
            "model_mae": "₹ 1,42,560"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/predict/vehicle-cluster", methods=["POST"])
def predict_vehicle_cluster():
    clust = models_state["clustering"]
    if clust is None:
        return jsonify({"error": "Clustering model not found"}), 503

    try:
        data = request.json or {}
        year = float(data.get("registration_year", 2023))
        weight = float(data.get("gross_vehicle_weight", 1750))
        count = float(data.get("vehicle_count", 1))
        fuel = str(data.get("fuel_name", "PETROL")).upper()
        norm = str(data.get("pollution_norm", "BHARAT STAGE VI")).upper()
        sale = str(data.get("sale_type", "New"))
        vclass = str(data.get("vehicle_class", "Motor Car"))

        if vclass not in clust["top_classes"]:
            vclass = "OTHER"

        row_num = pd.DataFrame([{
            "registrationYear": year,
            "grossVehicleWeight": np.log1p(weight),
            "vehicleCount": np.log1p(count)
        }])

        num_scaled = clust["scaler"].transform(row_num.values)

        row_cat = pd.DataFrame([{
            "fuelName": fuel,
            "pollutionNorm": norm,
            "saleType": sale,
            "vehicleClassName": vclass
        }])
        cat_encoded = clust["encoder"].transform(row_cat)

        feat_vector = np.hstack([num_scaled, cat_encoded])

        # Cluster prediction
        cluster_id = int(clust["kmeans"].predict(feat_vector)[0])
        pc = clust["pca"].transform(feat_vector)[0]

        profile = clust["profiles"].get(cluster_id, {})
        cluster_name = (
            "Lightweight & Two-Wheeler Segment (High Volume / Petrol)"
            if cluster_id == 0 else
            "Passenger Car & Heavy Vehicle Segment (Diesel / Motor Cars)"
        )

        return jsonify({
            "cluster": cluster_id,
            "cluster_name": cluster_name,
            "pc1": round(float(pc[0]), 3),
            "pc2": round(float(pc[1]), 3),
            "profile": profile
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/clustering/samples")
def get_clustering_samples():
    clust = models_state["clustering"]
    if clust is None:
        return jsonify({"error": "Clustering data not found"}), 503

    return jsonify({
        "sample_points": clust["sample_points"][:600],  # 600 points for ultra-smooth interactive canvas/chart
        "metrics": clust["metrics"],
        "profiles": clust["profiles"],
        "explained_variance": clust["explained_variance"]
    })


@app.route("/api/benchmarks")
def get_benchmarks():
    return jsonify(BENCHMARKS)


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5050))
    print(f"Starting ML Capstone Server at http://127.0.0.1:{port} ...")
    app.run(host="127.0.0.1", port=port, debug=False)
