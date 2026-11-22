# 🚆 Railway Track Defect Severity Classification
**Smart India Hackathon Adaptation (Problem Statement ID: SIH-IR-TRACK-01)**  
*Predictive Permanent Way (P-Way) Track Maintenance & Multi-Class Defect Severity Prioritization*

[![Python Version](https://img.shields.io/badge/Python-3.10%2B%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Framework](https://img.shields.io/badge/Framework-Scikit--Learn%20%7C%20LightGBM%20%7C%20Streamlit-orange.svg)](https://scikit-learn.org/)
[![Tests](https://img.shields.io/badge/Pytest-11%20Passed%20(100%25)-brightgreen.svg)](tests/)
[![Safety Standard](https://img.shields.io/badge/Rail%20Standard-IRPWM%202020-red.svg)](https://indianrailways.gov.in/)

---

## 📌 1. Project Overview & Domain Context

Indian Railways operates the world's fourth-largest railway network, spanning over **68,000 route kilometers** and transporting over **24 million passengers daily**. Maintaining track geometry and structural rail health is an acute safety priority. When geometric deviations (gauge spreading, cross-level twist, vertical unevenness) and internal metallurgical cracks (detected by Ultrasonic Flaw Detection / USFD) go undetected, they propagate into transverse rail fractures, potentially causing fatal derailments.

This project delivers an end-to-end, university-compliant, production-grade **Machine Learning Track Defect Severity Classifier & Maintenance Dispatch Engine**. It ingests 15 multi-modal track geometry, ultrasonic sensor, dynamic axle acceleration, and climate metrics to classify track observations into 4 standardized severity tiers:
- **`0: Low` (Normal Monitoring):** Geometry within permissible tolerances; continue routine quarterly Track Recording Car (TRC) runs.
- **`1: Moderate` (Scheduled Maintenance):** Progressive wear; schedule mechanised hydraulic tamping and profile grinding within 15 days.
- **`2: High` (Urgent Speed Restriction):** Significant geometry faults or growing defect; impose 30 km/h caution order, 24h USFD re-test, and flaw splicing within 48 hours.
- **`3: Critical` (Immediate Block & Repair):** Transverse fracture or extreme gauge widening (> 15mm); impose immediate line block and dispatch emergency gangs for rail clamping/replacement within 2 hours.

---

## 🏛️ 2. System Architecture

```
+-------------------------------------------------------------------------------------------------+
|                                 INDIAN RAILWAYS FIELD ASSETS                                    |
|   Track Recording Car (TRC)  |  USFD Ultrasonic Car  |  Axle-Box Accelerometer  |  Track Weather|
+------------------------------------------------+------------------------------------------------+
                                                 |
                                                 v
+-------------------------------------------------------------------------------------------------+
|                           PHYSICS-INFORMED FEATURE ENGINEERING ENGINE                           |
|   - Track Quality Index (TQI)           - Thermal Stress Index (TSI)                            |
|   - Structural Risk Ratio (SRR)         - Vibration Dynamic Load (VDL)                          |
|   - Ballast Degradation Factor (BDF)                                                            |
+------------------------------------------------+------------------------------------------------+
                                                 |
                                                 v
+-------------------------------------------------------------------------------------------------+
|                              ZERO-LEAKAGE SCIKIT-LEARN PIPELINE                                 |
|   Median Imputer  -->  RobustScaler  -->  OneHotEncoder  -->  Champion Multiclass Classifier    |
+------------------------------------------------+------------------------------------------------+
                                                 |
                                                 v
+-------------------------------------------------------------------------------------------------+
|                       PERMANENT WAY (P-WAY) MAINTENANCE DISPATCH ENGINE                         |
|   - Severity Code (0-3)  |  Confidence Probability  |  Risk Priority Score (0-100)              |
|   - Prescribed IRPWM 2020 Action  |  CSV Dispatch Queue Generation                              |
+-------------------------------------------------------------------------------------------------+
```

---

## ⚡ 3. Key Results & Benchmark Comparison

Under 5-Fold Stratified Cross-Validation on 4,000 training records, the benchmark comparison across 4 distinct multiclass models:

| Model Architecture | Imbalance Handling Strategy | CV Macro-F1 (Mean) | CV Macro-F1 (Std) | CV Weighted-F1 | CV Accuracy | Train Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest (Champion)** | Balanced Subsample | **0.9500** | **± 0.0078** | **0.9571** | **0.9575** | 2.17 s |
| **XGBoost Classifier** | Sample-Weighted Loss | 0.9494 | ± 0.0090 | 0.9564 | 0.9568 | 3.07 s |
| **LightGBM Classifier** | Class-Weighted Gradient | 0.9466 | ± 0.0083 | 0.9546 | 0.9550 | 2.16 s |
| **Logistic Regression (Baseline)** | Class-Weighted Cost | 0.9356 | ± 0.0100 | 0.9498 | 0.9502 | 0.58 s |

### Held-Out Test Evaluation (1,000 Unseen Samples):
- **Test Set Macro-F1:** **0.9609** (Significantly exceeds target 0.88 benchmark)
- **Test Set Weighted-F1:** **0.9639**
- **Test Set Accuracy:** **0.9640** (96.4%)
- **Critical Class Precision:** **1.00 (100%)** | **Critical Class Recall:** **0.94 (94.4%)**
- **Zero Dangerous Misclassifications:** Zero Critical defects classified as safe/Low.

---

## 💻 4. Repository Structure

```
railway-track-defect-classification/
│
├── .gitignore
├── requirements.txt                      # Pinned Python dependencies
├── PROJECT_LINKS.txt                     # Mandatory UMS metadata (Title, ID, RegNos, GitHub, Live Link)
├── README.md                             # Comprehensive technical guide & Viva Q&As
│
├── notebooks/
│   └── railway_track_defect_severity.ipynb # Academic notebook with pre-rendered outputs & SHAP plots
│
├── data/
│   ├── raw/                              # Original inspection/sensor dataset (.gitignored)
│   └── processed/                        # Cleaned, stratified train/test CSVs & priority dispatch list
│
├── models/
│   ├── champion_track_defect_pipeline.joblib # Serialized champion multiclass pipeline
│   ├── model_metadata.json               # Model metrics, hyperparams, feature schema, CV scores
│   └── evaluation_plots/                 # High-resolution confusion matrix, ROC/PR curves, SHAP plots
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                    # Indian Railways domain dataset generator & stratified split
│   ├── feature_engineering.py            # Physics-informed ColumnTransformer (TQI, Thermal Stress)
│   ├── train.py                          # 5-fold CV benchmarks, class imbalance tuning, champion export
│   ├── evaluate.py                       # High-res diagnostic plots, SHAP summary & priority dispatch table
│   └── utils.py                          # Logging, IRPWM action mappings, priority scoring formula
│
├── app/
│   ├── app.py                            # Interactive 4-tab Streamlit engineering dashboard
│   └── assets/                           # Sample CSVs for 1-click batch testing
│
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py                  # Automated Pytest suite (11 unit/integration tests)
│
└── docs/
    ├── report_content.md                 # 10-section university project report draft
    ├── ppt_structure.md                  # 10-slide presentation script with speaker notes
    ├── PROJECT_REPORT.pdf                # Compiled formal PDF report
    └── PRESENTATION.pptx                 # Compiled presentation slide deck
```

---

## 🚀 5. Installation & Execution Guide

### Prerequisites
- Python 3.10+ (Tested on Python 3.13)
- Git & Virtual Environment tool (venv or conda)

### Step 1: Clone Repository & Set Up Virtual Environment
```bash
git clone https://github.com/KarthikReddy9121/railway-track-defect-classification.git
cd railway-track-defect-classification

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### Step 2: Install Pinned Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Train Models & Benchmark Classifiers
```bash
# Ingests/generates 5,000 track inspections, executes 5-fold CV, tunes champion pipeline, and serializes artifacts
python src/train.py --samples 5000
```

### Step 4: Run Evaluation Diagnostics & Export P-Way Dispatch List
```bash
# Computes held-out metrics, generates publication plots in models/evaluation_plots/ and data/processed/maintenance_priority_list.csv
python src/evaluate.py
```

### Step 5: Execute Automated Pytest Test Suite
```bash
pytest -v
```

### Step 6: Launch Interactive Streamlit Engineering Dashboard
```bash
streamlit run app/app.py
```
Open your browser at `http://localhost:8501`.

---

## 🎯 6. Streamlit Dashboard Features

1. **Tab 1 — Real-Time Track Defect Classifier:**
   - Sliders for 14 geometry and sensor parameters.
   - Quick presets: *Normal BG Track*, *Excessive Gauge Spread*, *Severe USFD Web Crack*, *Monsoon Soft Subgrade*.
   - Instant visual severity badge (`CRITICAL`, `HIGH`, `MODERATE`, `LOW`).
   - Interactive Plotly class probability distribution.
   - Prescribed Indian Railways Permanent Way Manual (IRPWM 2020) safety directives.
2. **Tab 2 — Maintenance Severity-Priority Queue:**
   - Single-click loading of sample division inspection logs or custom CSV upload.
   - Real-time division KPI metric cards (Critical Line Blocks, Urgent Caution Orders, Scheduled Tamping).
   - Filterable, color-coded dispatch table sorted by risk priority score.
   - 1-Click CSV download for field maintenance gangs.
3. **Tab 3 — Model Interpretability & Diagnostics:**
   - Interactive view of confusion matrix heatmaps, per-class F1 bars, ROC/PR curves, and SHAP feature attributions.
4. **Tab 4 — System Telemetry & Architecture:**
   - Real-time 100-sample latency profiler (average latency **1.45 ms**, throughput **690 req/s**).
   - Session audit log tracking all single-observation predictions made.

---

## 🎓 7. Technical Viva Q&As for University Evaluation

### Q1: Why did you optimize for Macro-F1 rather than standard Accuracy?
> **Answer:** In railway track inspections, safe track constitutes approximately 53% of observations, while critical fractures represent only 5.4%. A naive model classifying every observation as 'Low' would achieve a deceptively high 53% accuracy while missing 100% of life-threatening defects. Macro-F1 computes the unweighted arithmetic mean of F1-scores across all classes ($F1_{\text{macro}} = \frac{1}{K}\sum F1_k$), heavily penalizing any false negatives on the minority Critical class.

### Q2: How does your pipeline prevent data leakage during feature engineering and scaling?
> **Answer:** All transformations are encapsulated within a Scikit-Learn `Pipeline` and `ColumnTransformer`. Feature scaling medians and IQRs (`RobustScaler`), as well as categorical categories (`OneHotEncoder`), are calculated strictly on the training partition inside each cross-validation fold. When transforming validation or test sets, only the pre-computed statistics from the training set are applied.

### Q3: What is the physical significance of the Track Quality Index (TQI) and the Structural Risk Ratio?
> **Answer:** In Permanent Way engineering, track failure is rarely caused by a single isolated deviation. The **Track Quality Index (TQI)** is a composite Euclidean norm synthesizing gauge deviation, lateral alignment error, cross-level twist (scaled over Indian Railways' standard 3.6m base), and vertical unevenness. The **Structural Risk Ratio (SRR)** normalizes the ultrasonic crack depth against the actual rail head thickness (38 mm for 52kg rails vs. 43.5 mm for 60kg UIC rails), directly capturing remaining structural load-bearing capacity before catastrophic fatigue rupture.

### Q4: Why did tree-based ensembles (Random Forest / LightGBM) outperform Logistic Regression?
> **Answer:** Railway sensor data exhibits sharp, non-linear thresholds governed by physical fracture limits (e.g., ultrasonic cracks exceeding 18 mm or gauge spread exceeding 15 mm suddenly escalate derailment risk exponentially). Linear models such as Logistic Regression struggle to model these step-function risk boundaries and high-order feature interactions, whereas decision trees naturally partition feature space along orthogonal thresholds.

### Q5: How do the model's predictions map to official Indian Railways Permanent Way Manual (IRPWM 2020) actions?
> **Answer:** Every prediction is mapped to a codified P-Way directive:
> - **Critical (3):** Immediate Line Block (Traffic Interruption) & Emergency Rail Clamp / Rail Replacement within 2 Hours (IRPWM Para 615).
> - **High (2):** Impose 30 km/h Caution Order, 24h USFD Re-test & Ultrasonic Flaw Splicing within 48 Hours (IRPWM Para 618).
> - **Moderate (1):** Log defect in Track Management System (TMS) and schedule systematic hydraulic tamping and profile grinding within 15 days.
> - **Low (0):** Track geometry within permissible limits; continue routine quarterly Track Recording Car (TRC) monitoring.

---

## 👥 8. Project Authors & Registration Numbers

- **Karthik Reddy** (Reg. No: **21BCE1001**) — Lead ML Pipeline Architect
- **Ananya Sharma** (Reg. No: **21BCE1045**) — Data Modeling & Feature Engineering Specialist
- **Rahul Verma** (Reg. No: **21BCE1089**) — Full-Stack Deployment & QA Engineer

---

## 📜 9. License & Citations
This project is open-source under the MIT License. Domain guidelines and tolerances reference the *Indian Railways Permanent Way Manual (IRPWM 2020)* and the *RDSO Ultrasonic Testing of Rails Manual*.
