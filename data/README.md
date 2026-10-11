# data/

Three datasets, one per track — all committed, so every notebook runs out of the box:

| File | Used by | Rows | Target |
|---|---|---|---|
| `raw/autos.csv` | Regression | ~371k | `price` (regression) |
| `raw/train.csv` | Classification | ~233k | `loan_default` (classification) |
| `raw/vehicle_registration/vehicle_registrations.csv` | Clustering | 500,000 | — (unsupervised) |

Both are committed directly to the repo (68MB and 40MB respectively, both under GitHub's 100MB limit) so every notebook runs out-of-the-box with no separate download step. `download_dataset.py` is kept as a fallback/reproducibility script for `autos.csv` specifically.

## Regression dataset: Used Cars

Source: [Kaggle](https://www.kaggle.com/datasets/thedevastator/uncovering-factors-that-affect-used-car-prices) — a cleaned republication of the eBay Kleinanzeigen (German eBay classifieds) used-car listings dataset. Each row is one used-car listing (~371k rows raw, ~312k after cleaning).

**Data dictionary:**

| Column | Description |
|---|---|
| `dateCrawled` | When the ad was first crawled |
| `name` | Name/title of the car listing |
| `seller` | Whether the seller is private or a dealer |
| `offerType` | Type of listing (offer/request) |
| `price` | Listed price of the car — **regression target** |
| `abtest` | Whether the listing was part of an A/B test |
| `vehicleType` | Vehicle body type (e.g. sedan, SUV, bus) |
| `yearOfRegistration` | Year the car was first registered |
| `gearbox` | Transmission type (manual/automatic) |
| `powerPS` | Engine power, in PS (metric horsepower) |
| `model` | Car model name |
| `kilometer` | Kilometers driven (odometer reading) |
| `monthOfRegistration` | Month the car was first registered |
| `fuelType` | Fuel type (petrol, diesel, LPG, etc.) |
| `brand` | Car manufacturer/brand |
| `notRepairedDamage` | Whether the car has unrepaired damage |
| `dateCreated` | Date the ad was created on the site |
| `nrOfPictures` | Number of pictures in the listing |
| `postalCode` | Postal code of the seller |
| `lastSeen` | Last time the crawler saw this ad online |

**Cleaning applied:** filter to realistic `price` (€100–150,000), `yearOfRegistration` (1950–2016), and `powerPS` (1–1000); drop duplicates and non-predictive columns (IDs, timestamps, picture count); engineer `vehicle_age = 2016 - yearOfRegistration`. Subsampled to 8,000 rows for computational feasibility of SVR/GridSearchCV.

**Re-fetching it (if ever needed):**
```bash
pip install kaggle
# place your Kaggle API token at ~/.kaggle/kaggle.json (see download_dataset.py docstring)
python data/download_dataset.py
```

## Classification dataset: Vehicle Loan Default Prediction

Source: [Kaggle](https://www.kaggle.com/datasets/mamtadhaker/lt-vehicle-loan-default-prediction) (L&T Financial Services) — borrower demographics, loan details, and credit-bureau history for ~233k vehicle loans, with `loan_default` (1 = defaulted, 0 = did not) as a binary target. Deliberately a different, harder problem from the regression track — realistic, imbalanced (~78%/22%) credit-risk classification.

**Data dictionary (key columns — see the notebook for the full list):**

| Column | Description |
|---|---|
| `disbursed_amount` | Loan amount disbursed |
| `asset_cost` | Cost of the financed vehicle |
| `ltv` | Loan-to-value ratio (%) |
| `Date.of.Birth` / `DisbursalDate` | Used to derive `age_years` in preprocessing |
| `Employment.Type` | Salaried / Self-employed (has missing values) |
| `PERFORM_CNS.SCORE` | Credit bureau score (0 = no bureau history, see `PERFORM_CNS.SCORE.DESCRIPTION`) |
| `PERFORM_CNS.SCORE.DESCRIPTION` | Risk-bucket label for the bureau score |
| `PRI.NO.OF.ACCTS` / `PRI.ACTIVE.ACCTS` / `PRI.OVERDUE.ACCTS` | Primary credit account counts |
| `PRI.CURRENT.BALANCE`, `PRIMARY.INSTAL.AMT` | Primary account balance / instalment amount |
| `NEW.ACCTS.IN.LAST.SIX.MONTHS`, `DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS` | Recent credit behaviour |
| `AVERAGE.ACCT.AGE`, `CREDIT.HISTORY.LENGTH` | `"Xyrs Ymon"` strings — parsed to months in preprocessing |
| `NO.OF_INQUIRIES` | Number of credit inquiries |
| `manufacturer_id`, `State_ID` | Low-cardinality categorical IDs (kept); `branch_id`/`supplier_id`/`Current_pincode_ID`/`Employee_code_ID` are dropped (too high-cardinality to one-hot encode) |
| `loan_default` | **Classification target** (1 = default, 0 = no default) |

**Cleaning/feature engineering applied:** `age_years` derived from `Date.of.Birth`/`DisbursalDate` (with a century-window fix — some 2-digit birth years parse to the 2060s otherwise); `AVERAGE.ACCT.AGE`/`CREDIT.HISTORY.LENGTH` parsed from `"Xyrs Ymon"` strings into total months; high-cardinality ID columns dropped. Subsampled to 8,000 rows for computational feasibility of SVC on this environment's hardware; stratified train/test split given the class imbalance.

## Clustering dataset: Indian Vehicle Registration Data (2020–25)

Source: [Kaggle](https://www.kaggle.com/datasets/aatifahmad123/indian-vehicle-registration-data-202025) (VAHAN registration records). Committed as `raw/vehicle_registration/vehicle_registrations.csv` (82 MB, 500,000 rows × 15 columns). Each row is a **group** of vehicles registered in one month at one RTO with the same maker, model, fuel, class and norm; `vehicleCount` is the group size.

| Column | Description | Used in clustering |
|---|---|---|
| `registrationYear` | Year of registration (2020–25; 5 impossible values such as 9100 are removed) | numeric feature |
| `financialYear`, `registrationMonthMMYY` | Financial year / month of registration | no (time stamps; used in EDA) |
| `makerName`, `vehicleModelName` | Manufacturer / model (1,244 and 14,160 values) | no (too high-cardinality) |
| `stateName`, `rtoCode`, `rtoName` | Registering state and RTO | no (geography; used in EDA) |
| `vehicleCategoryName` | Official category (16 values, e.g. LIGHT MOTOR VEHICLE, TWO WHEELER(NT)) | **held out** — post-hoc validation only |
| `vehicleClassName` | Vehicle class (67 values; top 12 kept, rest `OTHER`) | categorical feature |
| `fuelName` | Fuel type (21 values) | categorical feature |
| `pollutionNorm` | Emission norm (24 values, e.g. BHARAT STAGE VI) | categorical feature |
| `saleType` | `New` / `Resale` | categorical feature |
| `grossVehicleWeight` | Gross vehicle weight in kg (0 = missing in 4.8 % of rows) | numeric feature (`0`→missing, `log1p`) |
| `vehicleCount` | Vehicles in the group (median 1, max 2,078) | numeric feature (`log1p`) |

**Cleaning / preprocessing (see `notebooks/clustering.ipynb`, Section B):** keep `registrationYear` 2020–2025, treat weight 0 as missing (median imputation), `log1p` on weight and count, one-hot encode the four categorical features, standardise the numeric features only, fixed-seed 4,000-record subsample (Agglomerative clustering is O(n²) in memory). `download_dataset.py vehicle-reg` re-fetches the file from Kaggle if needed.
