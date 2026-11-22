# 🚆 ACADEMIC PROJECT REPORT: RAILWAY TRACK DEFECT SEVERITY CLASSIFICATION
**Smart India Hackathon Theme Adaptation (Problem Statement ID: SIH-IR-TRACK-01)**  
*Predictive Permanent Way (P-Way) Track Maintenance & Multi-Class Defect Severity Prioritization*

---

## 1. Title Page

- **Project Title:** Railway Track Defect Severity Classification
- **Problem Statement ID:** SIH-IR-TRACK-01 (Smart India Hackathon / Indian Railways Track Safety Theme)
- **Domain / Subject:** Supervised Machine Learning / Predictive Maintenance & Rail Safety Engineering
- **Target Safety Standard:** Indian Railways Permanent Way Manual (IRPWM 2020) & USFD Testing Standards
- **Academic Semester / Year:** University B.Tech / M.Tech Capstone Project Evaluation (UMS-Compliant)

### Evaluated Project Team:
| S.No | Student Name | Registration Number | Academic Role & Primary Contribution |
| :--- | :--- | :--- | :--- |
| 1 | **Karthik Reddy** | **21BCE1001** | **Lead ML Pipeline Architect:** Problem formulation, data pipeline design, zero-leakage ColumnTransformer, multi-model cross-validation benchmarking, and champion model serialization. |
| 2 | **Ananya Sharma** | **21BCE1045** | **Data Modeling & Feature Engineering Specialist:** Indian Railways domain feature design (TQI, Thermal Stress, Structural Risk Ratio), class imbalance strategy, hyperparameter optimization, and SHAP explainability. |
| 3 | **Rahul Verma** | **21BCE1089** | **Full-Stack Deployment & Quality Assurance Engineer:** Streamlit dashboard implementation (4 tabs, dual inspection modes), P-Way dispatch queue ranking, automated Pytest test suite, and technical documentation. |

---

## 2. Introduction and Problem Statement

### 2.1 Domain Background
Indian Railways (IR) is the fourth-largest national railway network in the world by size, comprising over **68,000 route kilometers** of broad gauge tracks, operating more than **13,000 passenger trains** and **9,000 freight trains daily**, and transporting over **24 million passengers per day**. With track traffic density continuously escalating—frequently exceeding **60 to 100 Gross Million Tonnes (GMT) per annum** on major trunk corridors (such as the Golden Quadrilateral and Dedicated Freight Corridors)—the physical infrastructure experiences relentless dynamic cyclic loading and severe thermal fluctuations.

The structural integrity of railway track depends on strict adherence to geometry standards (gauge, cross-level twist, vertical unevenness, and alignment) as well as the metallurgical soundess of rail steel. When micro-defects or geometric irregularities remain undetected or unaddressed, they propagate into transverse rail fractures, gauge spread, rail buckling, or wheel climb, potentially resulting in catastrophic derailments.

### 2.2 Current Operational Shortcomings
Under traditional Indian Railways Permanent Way (P-Way) maintenance workflows:
1. **Periodic Rather Than Condition-Based:** Track Recording Cars (TRC) operate on fixed quarterly or semi-annual cycles, while Ultrasonic Flaw Detection (USFD) teams conduct manual hand-pushed trolley tests at intervals of 1 to 3 months depending on GMT.
2. **Binary or Fragmented Inspection Silos:** Geometry recording (TRC) and ultrasonic internal flaw data (USFD) are analyzed in separate administrative silos without unified mathematical synthesis.
3. **Severe Class Imbalance Dilemma:** Real-world railway tracks exhibit an inherently skewed defect distribution: approximately **55%** of track segments exhibit normal permissible geometry (Low severity), **25%** exhibit moderate progressive wear (Moderate severity), **14%** present significant geometry deviations requiring urgent speed restrictions (High severity), and only **5-6%** present acute, life-threatening rail cracks or extreme gauge widening (Critical severity). Standard off-the-shelf classifiers tend to maximize overall accuracy by overfitting to the majority `Low` class while failing to detect deadly `Critical` defects.
4. **Lack of Risk-Weighted Maintenance Prioritization:** Senior Section Engineers (SSE/Track) receive raw, non-prioritized defect logs numbering in the thousands. Without an automated, traffic-calibrated severity-priority scoring system, field gangs cannot optimize their emergency track possessions (line blocks).

### 2.3 Proposed Solution: SIH-IR-TRACK-01 System
This project engineers a machine-learning-driven **Track Defect Severity Classification & Maintenance Dispatch Engine**. By assimilating 15 multi-modal track geometry, ultrasonic sensor, dynamic axle acceleration, and environmental attributes, the system:
- Classifies track segments into 4 standardized severity tiers: **Low, Moderate, High, and Critical**.
- Optimizes strictly for the **Macro-Averaged F1 Score** to ensure maximum recall on minority critical defects.
- Generates an actionable, traffic-weighted **Maintenance Priority Score (0-100)** mapped directly to Indian Railways Permanent Way Manual (IRPWM 2020) safety directives.

---

## 3. Project Objectives and Need

### 3.1 Primary Project Objectives
1. **Develop a Multi-Class Classifier Calibrated for Severe Class Imbalance:** Benchmark and tune state-of-the-art algorithms (Logistic Regression, Random Forest, XGBoost, LightGBM) to achieve a **Macro-F1 score exceeding 0.90** on minority Critical and High severity defects.
2. **Formulate Physics-Informed Domain Features:** Engineer composite indices reflecting rail physics, including the Track Quality Index (TQI), Thermal Stress Index (TSI), Structural Risk Ratio (SRR), Vibration Dynamic Load (VDL), and Ballast Degradation Factor (BDF).
3. **Guarantee Zero Data Leakage:** Encapsulate all transformations (numerical imputation, robust scaling, categorical one-hot encoding) inside an isolated Scikit-Learn `ColumnTransformer` and `Pipeline` fitted strictly on training partitions.
4. **Construct an Operational Maintenance Priority Queue:** Create an operational dispatch algorithm that weights predicted severity, classification confidence, and route traffic density (GMT) to output sorted dispatch orders for P-Way gangs.
5. **Deploy an Interactive Engineering Dashboard:** Provide an accessible Streamlit interface supporting single-observation real-time diagnosis, batch CSV upload, interactive SHAP interpretability, and system latency telemetry.
6. **Ensure Strict University & Industry Verification:** Achieve 100% test pass rate across an automated Pytest test suite validating data schema, probability calibration, edge-case safety, and model serialization.

### 3.2 Societal & Industrial Need
- **Preventing Loss of Human Life:** Early detection of transverse fissures and gauge spread directly mitigates the risk of catastrophic passenger train derailments.
- **Operational P-Way Efficiency:** Eliminates manual triage of inspection reports, allowing Indian Railways divisions to deploy maintenance gangs to high-risk zones during scarce 2-hour traffic blocks.
- **Asset Life Extension:** Predictive tamping and rail profile grinding arrest premature rail head degradation, saving hundreds of crores in unnecessary premature rail replacement.

---

## 4. Project Overview, Key Features, and Working

### 4.1 System Overview
The system processes continuous Track Recording Car (TRC) geometry telemetry, USFD ultrasonic echographic defect data, axle-box vertical accelerations, rail temperatures, and monsoon rainfall metrics across Indian Railways divisions.

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

### 4.2 Key Features
1. **Multi-Modal Track Ingestion:** Ingests 15 parameters spanning logistics, geometry, ultrasonics, dynamic forces, and climate.
2. **Physics-Consistent Data Synthesis:** Realistic Indian Railways data generator reflecting the actual Broad Gauge (1676 mm) geometry limits and IRPWM tolerances.
3. **Class-Weighted Imbalance Handling:** Uses balanced class weighting and sample weighting to penalize minority misclassifications.
4. **Automated Risk Scoring & Action Mapping:** Automatically maps predictions to formal IRPWM directives:
   - **Critical (3):** Immediate Line Block (Traffic Interruption) & Emergency Rail Clamp / Rail Replacement within 2 Hours (IRPWM Para 615).
   - **High (2):** Impose 30 km/h Caution Order, 24h USFD Re-test & Ultrasonic Flaw Splicing within 48 Hours (IRPWM Para 618).
   - **Moderate (1):** Schedule Systematic Hydraulic Tamping & Rail Profile Grinding within 15 Days.
   - **Low (0):** Normal Track Geometry — Continue Routine Quarterly TRC Monitoring.
5. **Interactive Dual-Mode Streamlit Dashboard:** Single inspection simulation with real-world presets and high-throughput batch CSV processing.
6. **Local & Global SHAP Interpretability:** TreeExplainer attributions visually explaining which physical thresholds trigger Critical classifications.

---

## 5. Tools, Technologies, and Frameworks Used

| Category | Tool / Library | Version | Technical Justification |
| :--- | :--- | :--- | :--- |
| **Programming Language** | Python | 3.10+ / 3.13 | Universal standard for scientific ML, rapid prototyping, and high-performance libraries. |
| **Core Scientific Libraries** | NumPy & Pandas | Latest | High-speed vectorized array operations, data manipulation, and tabular schema validation. |
| **Machine Learning Core** | Scikit-learn | >= 1.4.0 | Provides `ColumnTransformer`, `Pipeline`, cross-validation routines, and baseline models. |
| **Imbalance Handling** | Imbalanced-learn | >= 0.12.0 | Advanced class-weight computation, SMOTE resampling, and balanced subsampling estimators. |
| **Gradient Boosted Trees** | LightGBM & XGBoost | Latest | Optimized tree-boosting algorithms providing high sample-weighted classification performance and low inference latency. |
| **Model Interpretability** | SHAP | >= 0.54.0 | Game-theoretic Shapley additive feature importance explaining individual and global predictions. |
| **Plotting & Diagnostics** | Matplotlib & Seaborn | Latest | Publication-quality static evaluation plots (Confusion matrix heatmaps, ROC/PR curves). |
| **Interactive Visualization** | Plotly | >= 5.18.0 | Interactive web-based probability distribution charts embedded in the Streamlit application. |
| **Web Application** | Streamlit | >= 1.32.0 | Reactive, production-ready engineering dashboard requiring no complex frontend infrastructure. |
| **Model Serialization** | Joblib | >= 1.3.0 | High-performance serialization for large NumPy/Scikit-learn pipeline objects. |
| **Quality Assurance** | Pytest | >= 8.0.0 | Enterprise test runner verifying data integrity, zero leakage, and edge-case safety. |

---

## 6. Methodology and Implementation

### 6.1 Detailed Dataset Description
The dataset models Indian Railways broad gauge (1676 mm) permanent way infrastructure:
- **Total Records:** 5,000 inspection instances.
- **Feature Space:** 14 independent features (11 numerical, 3 categorical).
- **Target Variable:** `severity_level` (0: Low, 1: Moderate, 2: High, 3: Critical).
- **Class Proportions:** 52.8% Low (2,640 records), 27.5% Moderate (1,375 records), 14.3% High (715 records), 5.4% Critical (270 records).

#### Detailed Feature Dictionary:
1. `observation_id` (String): Unique identifier (e.g., `IR-TRC-2024-00042`).
2. `zone_division` (Categorical): Operating railway zone (`NR-UMB`, `SR-MAS`, `CR-BB`, `ER-HWH`, `SCR-SC`, `WR-BCT`, `NCR-PRYJ`, `ECoR-KUR`).
3. `track_km_post` (Numerical): Track chainage in kilometers (5.0 to 950.0 km).
4. `rail_section_type` (Categorical): Rail weight (`52kg` or `60kg UIC`).
5. `sleeper_type` (Categorical): Sleeper support (`PSC` Pre-stressed Concrete, `Steel`, `Wooden`).
6. `traffic_density_gmt` (Numerical): Annual traffic load in Gross Million Tonnes (5.0 to 115.0 GMT).
7. `gauge_deviation_mm` (Numerical): Deviation from standard 1676 mm gauge (-5.5 mm to +28.0 mm).
8. `alignment_error_mm` (Numerical): Lateral alignment error (0.1 mm to 24.0 mm).
9. `twist_mm_per_m` (Numerical): Cross-level twist measured across 3.6m base (0.1 to 9.5 mm/m).
10. `unevenness_mm` (Numerical): Longitudinal vertical surface irregularity (0.5 to 26.0 mm).
11. `ultrasonic_flaw_depth_mm` (Numerical): Echographic flaw depth in rail head/web (0.0 to 35.0 mm).
12. `axle_box_accel_g` (Numerical): Peak vertical dynamic acceleration (0.04g to 1.45g).
13. `rail_temp_celsius` (Numerical): Rail steel temperature (-4°C to 71°C).
14. `rainfall_7d_mm` (Numerical): 7-day cumulative rainfall (0.0 to 320.0 mm).
15. `days_since_last_tamping` (Numerical): Elapsed days since last mechanised tamping (5 to 600 days).

### 6.2 Preprocessing & Data Cleaning
- **Boundary Clamping:** Unphysical sensor anomalies are bounded to realistic railway limits (e.g., gauge deviation clamped within $[-10, +35]\text{ mm}$, flaw depth clamped within $[0, 45]\text{ mm}$, axle acceleration clamped within $[0.01, 2.5]\text{ g}$).
- **Missing Value Strategy:** Missing numerical values are imputed using median values computed strictly on the training set (`SimpleImputer(strategy='median')`). Missing categorical values are imputed using mode (`SimpleImputer(strategy='most_frequent')`).
- **Robust Scaling:** `RobustScaler` scales numerical features using median and interquartile range (IQR), preventing extreme sensor outliers (e.g., deep cracks or sudden shock accelerations) from distorting scaling distributions.
- **One-Hot Encoding:** Categorical attributes (`zone_division`, `rail_section_type`, `sleeper_type`) are encoded using `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`.

### 6.3 Physics-Informed Feature Engineering Formulations
1. **Track Quality Index (TQI):**
   $$\text{TQI} = \sqrt{(\text{gauge\_deviation\_mm})^2 + (\text{alignment\_error\_mm})^2 + (\text{twist\_mm\_per\_m} \times 3.6)^2 + (\text{unevenness\_mm})^2}$$
2. **Thermal Stress Index (TSI):**
   $$\text{TSI} = |\text{rail\_temp\_celsius} - 40.0| \times \left(\frac{\text{traffic\_density\_gmt}}{50.0}\right)$$
3. **Structural Risk Ratio (SRR):**
   $$\text{SRR} = \frac{\text{ultrasonic\_flaw\_depth\_mm}}{\text{Rail Head Depth (38.0mm for 52kg, 43.5mm for 60kg)}}$$
4. **Vibration Dynamic Load (VDL):**
   $$\text{VDL} = \text{axle\_box\_accel\_g} \times \left(\frac{\text{traffic\_density\_gmt}}{40.0}\right)$$
5. **Ballast Degradation Factor (BDF):**
   $$\text{BDF} = \left(\frac{\text{days\_since\_last\_tamping}}{100.0}\right) \times \left(1.0 + \frac{\text{rainfall\_7d\_mm}}{100.0}\right)$$

### 6.4 Model Architectures Explored & Justification
- **Logistic Regression (Class-Weighted):** Serves as our linear, interpretable baseline. Penalizes errors inversely proportional to class frequencies using `class_weight='balanced'`.
- **Random Forest Classifier (Balanced Subsample):** Builds 150 randomized decision trees where each bootstrap sample computes balanced weights on the fly (`class_weight='balanced_subsample'`). Captures non-linear thresholds with high stability.
- **XGBoost Classifier:** Tree boosting optimizing multi-class log-loss with sample weights computed inversely from class frequencies. Highly effective at gradient step correction.
- **LightGBM Classifier:** Leaf-wise tree boosting with `class_weight='balanced'`. Achieves near-instantaneous inference latency (< 5 ms per sample).

### 6.5 Evaluation Metrics Justification
- **Why Accuracy Fails:** In our dataset with 52.8% Low defects, a model that classifies everything as `Low` would report a deceptively high 52.8% accuracy while missing 100% of Critical defects, causing fatal derailments.
- **Macro-Averaged F1 Score:** Weights every class equally regardless of sample frequency:
  $$\text{Macro-F1} = \frac{\text{F1}_{\text{Low}} + \text{F1}_{\text{Moderate}} + \text{F1}_{\text{High}} + \text{F1}_{\text{Critical}}}{4}$$
  A drop in recall for the minority `Critical` class immediately penalizes the Macro-F1 score severely.
- **Per-Class Recall on Critical ($R_{\text{Critical}}$):** Evaluates the exact safety requirement: what percentage of actual rail fractures were successfully flagged?

### 6.6 Comprehensive Model Comparison Table
*(Evaluated under 5-Fold Stratified Cross-Validation on the 4,000-sample training partition)*

| Model Architecture | Imbalance Handling Strategy | CV Macro-F1 (Mean) | CV Macro-F1 (Std) | CV Weighted-F1 | CV Accuracy | Train Latency (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Random Forest (Champion)** | Balanced Subsample | **0.9500** | **± 0.0078** | **0.9571** | **0.9575** | 2.17 s |
| **XGBoost Classifier** | Sample-Weighted Loss | 0.9494 | ± 0.0090 | 0.9564 | 0.9568 | 3.07 s |
| **LightGBM Classifier** | Class-Weighted Gradient | 0.9466 | ± 0.0083 | 0.9546 | 0.9550 | 2.16 s |
| **Logistic Regression (Baseline)** | Class-Weighted Cost | 0.9356 | ± 0.0100 | 0.9498 | 0.9502 | 0.58 s |

### 6.7 Priority Scoring & Dispatch Ranking Engine
To translate class predictions into operational P-Way field commands, we compute a **Maintenance Priority Score ($S_{\text{priority}} \in [0, 100]$)**:
$$S_{\text{priority}} = \text{clip}\left( \left( w_k \cdot 15.0 \cdot P(\hat{y}=k) \cdot \left(1.0 + \text{clip}\left(\frac{\text{GMT}}{80.0}, 0.1, 1.5\right)\right) \right) + 0.8 \cdot \min(\text{flaw}, 25.0), \ 0.0, \ 100.0 \right)$$
Where severity weights are:
- $w_{\text{Critical}} = 5.0$
- $w_{\text{High}} = 3.0$
- $w_{\text{Moderate}} = 1.75$
- $w_{\text{Low}} = 1.0$

This formula ensures that high-traffic routes (> 80 GMT) escalate in maintenance priority due to rapid dynamic cyclic fatigue, and deep ultrasonic cracks receive an immediate score escalation.

---

## 7. Results, Testing, and Relevant Screenshots

### 7.1 Held-Out Test Evaluation Results (1,000 Samples)
The champion Random Forest pipeline was evaluated on the unseen held-out test partition:
- **Test Macro-F1 Score:** **0.9609** (Substantially exceeds target threshold of 0.88)
- **Test Weighted-F1 Score:** **0.9639**
- **Test Overall Accuracy:** **0.9640** (96.4%)

#### Detailed Per-Class Performance Table:
| Defect Severity Tier | Precision | Recall (Sensitivity) | F1-Score | Test Support |
| :--- | :--- | :--- | :--- | :--- |
| **Low** (Routine Monitoring) | 0.97 | 0.99 | 0.98 | 528 |
| **Moderate** (Scheduled Maintenance) | 0.94 | 0.93 | 0.94 | 275 |
| **High** (30 km/h Speed Restriction) | 0.96 | 0.94 | 0.95 | 143 |
| **Critical** (Immediate Line Block) | **1.00** | **0.94** | **0.97** | 54 |
| **Macro Average** | **0.97** | **0.95** | **0.96** | 1000 |
| **Weighted Average** | **0.96** | **0.96** | **0.96** | 1000 |

### 7.2 Confusion Matrix Diagnostics
The normalized confusion matrix demonstrates that **94.4% of Critical defects** and **94.4% of High defects** are correctly classified. Furthermore, **zero** Critical defects were misclassified as Low, guaranteeing that no dangerous rail fractures are classified as safe.

### 7.3 Automated Pytest Verification Summary
The project contains an automated Pytest test suite (`tests/test_pipeline.py`) verifying 11 distinct test cases:
```
tests/test_pipeline.py::test_dataset_generation_schema PASSED            [  9%]
tests/test_pipeline.py::test_class_imbalance_distribution PASSED         [ 18%]
tests/test_pipeline.py::test_data_cleaning_and_bounds PASSED             [ 27%]
tests/test_pipeline.py::test_domain_feature_engineer PASSED              [ 36%]
tests/test_pipeline.py::test_zero_data_leakage_isolation PASSED          [ 45%]
tests/test_pipeline.py::test_pipeline_prediction_probabilities PASSED    [ 54%]
tests/test_pipeline.py::test_edge_case_pristine_track PASSED             [ 63%]
tests/test_pipeline.py::test_edge_case_critical_transverse_crack PASSED  [ 72%]
tests/test_pipeline.py::test_priority_score_ranking_consistency PASSED   [ 81%]
tests/test_pipeline.py::test_maintenance_dispatch_table_generation PASSED [ 90%]
tests/test_pipeline.py::test_export_artifacts_exist PASSED               [100%]
============================= 11 passed in 2.30s ==============================
```

### 7.4 Streamlit Engineering Dashboard Verification
The application provides four operational tabs:
1. **Tab 1 — Real-Time Track Defect Classifier:** Sliders for geometry and sensor values, preset buttons, instant severity badges (`CRITICAL`, `HIGH`, `MODERATE`, `LOW`), Plotly probability distributions, and IRPWM directives.
2. **Tab 2 — Maintenance Severity-Priority Queue:** Batch inspection CSV upload, division KPIs, color-coded ranked data tables, and downloadable dispatch CSVs.
3. **Tab 3 — Model Interpretability & Diagnostics:** Displays confusion matrix heatmaps, per-class F1 metrics, ROC/PR curves, and SHAP TreeExplainer attributions.
4. **Tab 4 — System Telemetry & Architecture:** Real-time 100-sample latency profiler (average latency **1.45 ms** per sample, throughput **690 req/s**), and session audit log.

---

## 8. Individual Contributions of All Group Members

To ensure complete transparency and evaluable accountability, responsibilities were divided as follows:

### 1. Karthik Reddy (Reg. No: 21BCE1001) — Lead Machine Learning Engineer & Pipeline Architect
- **Data Pipeline & Schema Design:** Engineered the Indian Railways domain dataset generator in `src/data_loader.py` with physical bounds for Broad Gauge (1676 mm).
- **Zero-Leakage Architecture:** Formulated the Scikit-Learn `ColumnTransformer` and pipeline encapsulation in `src/feature_engineering.py`.
- **Model Benchmarking & Serialization:** Implemented 5-Fold Stratified Cross-Validation across Logistic Regression, Random Forest, XGBoost, and LightGBM in `src/train.py`, optimizing for Macro-F1 and serializing champion pipeline artifacts to `models/champion_track_defect_pipeline.joblib`.
- **Academic Notebook Leadership:** Co-authored and executed the comprehensive academic Jupyter Notebook (`notebooks/railway_track_defect_severity.ipynb`).

### 2. Ananya Sharma (Reg. No: 21BCE1045) — Data Modeling & Feature Engineering Specialist
- **Physics-Informed Domain Engineering:** Formulated the mathematical equations for Track Quality Index (TQI), Thermal Stress Index (TSI), Structural Risk Ratio (SRR), and Ballast Degradation Factor (BDF).
- **Class Imbalance Strategy:** Evaluated balanced class weights versus SMOTE oversampling to eliminate minority recall degradation.
- **Model Interpretability (SHAP):** Conducted TreeExplainer SHAP attribution analysis in `src/evaluate.py`, identifying flaw depth and gauge spread as the dominant failure drivers.
- **Diagnostic Plot Export:** Authored routines generating publication-grade confusion matrices, per-class F1 bar charts, and multiclass ROC/PR curves.

### 3. Rahul Verma (Reg. No: 21BCE1089) — Full-Stack Deployment & Quality Assurance Engineer
- **Streamlit Engineering Dashboard:** Designed and implemented the complete 4-tab interactive dashboard in `app/app.py` with custom Indian Railways branding, presets, and Plotly graphics.
- **P-Way Dispatch Queue Logic:** Implemented the Maintenance Priority Score formula and dispatch ranking table in `src/utils.py`.
- **Automated Test Suite:** Created the 11-test Pytest suite in `tests/test_pipeline.py` verifying data schema, leakage isolation, probability calibration, and edge cases.
- **Technical Documentation & Presentation:** Authored the comprehensive 10-section report (`docs/report_content.md`), presentation slides script (`docs/ppt_structure.md`), and compiled PDF/PPTX deliverables.

---

## 9. Limitations and Future Scope

### 9.1 Limitations
1. **Sensor Calibration Drift:** Assumes sensor telemetry from Track Recording Cars and USFD trolleys is calibrated to IRPWM tolerances; extreme field sensor drift could introduce measurement noise.
2. **Subgrade Geological Variability:** Soil strata characteristics (e.g., black cotton soil swelling versus rocky ballast subgrade) are currently represented through rainfall and tamping age proxies rather than direct geotechnical borehole sensors.
3. **Synchronous Single-Node Processing:** The current Streamlit application processes CSV batches in-memory; scaling to network-wide multi-terabyte continuous video inspection streams requires distributed processing.

### 9.2 Future Scope & Next-Phase Roadmap
1. **Integration with Indian Railways Track Management System (TMS):** Directly interface via REST APIs to ingest live TRC and USFD data streams from CRIS (Centre for Railway Information Systems).
2. **IoT Rail-Mounted Vibration Accelerometers:** Incorporate real-time LoRaWAN-connected vibration nodes installed at high-risk switch points and diamond crossings.
3. **Satellite InSAR Subgrade Subsidence Tracking:** Integrate European Space Agency (Sentinel-1) Interferometric Synthetic Aperture Radar data to monitor track embankment settlement along coastal monsoon routes.
4. **Computer Vision Rail Surface Defect Detection:** Add a YOLOv11 / Vision Transformer sidecar to classify surface squats, wheel burns, and spalling from high-speed undercarriage camera feeds.

---

## 10. References and Resources Used

1. **Indian Railways Permanent Way Manual (IRPWM 2020):** Ministry of Railways, Railway Board, Government of India. Chapter VI: Track Monitoring and Track Maintenance Standards; Chapter IX: Ultrasonic Flaw Detection (USFD).
2. **Manual for Ultrasonic Testing of Rails and Welded Joints (2020 Edition):** Research Designs and Standards Organisation (RDSO), Ministry of Railways, Lucknow.
3. **Lundqvist, A., & Dahlberg, T. (2005):** "Load impact on railway track due to wheel flat." *Proceedings of the Institution of Mechanical Engineers, Part F: Journal of Rail and Rapid Transit*, 219(2), 75-87.
4. **Pedregosa, F., et al. (2011):** "Scikit-learn: Machine Learning in Python." *Journal of Machine Learning Research*, 12, 2825-2830.
5. **Lundberg, S. M., & Lee, S. I. (2017):** "A Unified Approach to Interpreting Model Predictions." *Advances in Neural Information Processing Systems (NeurIPS 2017)*, 30, 4765-4774.
6. **Chen, T., & Guestrin, C. (2016):** "XGBoost: A Scalable Tree Boosting System." *ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794.
7. **Ke, G., et al. (2017):** "LightGBM: A Highly Efficient Gradient Boosting Decision Tree." *Advances in Neural Information Processing Systems (NeurIPS 2017)*, 3146-3154.
8. **Chawla, N. V., et al. (2002):** "SMOTE: Synthetic Minority Over-sampling Technique." *Journal of Artificial Intelligence Research*, 16, 321-357.
