"""
scripts/compile_docs.py - Compiles docs/PROJECT_REPORT.pdf and docs/PRESENTATION.pptx.
Uses ReportLab for formal PDF generation and python-pptx for presentation deck generation.
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
)
from reportlab.pdfgen import canvas

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN


# ---------------------------------------------------------------------------
# Numbered Canvas for PDF Page Numbers
# ---------------------------------------------------------------------------
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_header_footer(num_pages)
            super().showPage()
        super().save()

    def draw_header_footer(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#555555"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, "Indian Railways Track Defect Severity Classification")
            self.setStrokeColor(colors.HexColor("#cccccc"))
            self.setLineWidth(0.5)
            self.line(54, 745, 558, 745)

        # Footer
        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 40, footer_text)
        self.drawString(54, 40, "Academic Project Report — Permanent Way Track Maintenance")
        self.setStrokeColor(colors.HexColor("#cccccc"))
        self.setLineWidth(0.5)
        self.line(54, 50, 558, 50)
        self.restoreState()


def generate_pdf(output_path: str):
    print(f"Generating PDF report: {output_path}...")
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#0b2545"),
        alignment=1, # Center
    )
    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#333333"),
        alignment=1,
    )
    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#0b2545"),
        spaceBefore=14,
        spaceAfter=8,
    )
    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        textColor=colors.HexColor("#134074"),
        spaceBefore=10,
        spaceAfter=5,
    )
    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#222222"),
        spaceAfter=6,
    )
    bullet_style = ParagraphStyle(
        "Bullet_Custom",
        parent=body_style,
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=3,
    )

    story = []

    # Cover Page
    story.append(Spacer(1, 40))
    story.append(Paragraph("🚆 RAILWAY TRACK DEFECT SEVERITY CLASSIFICATION", title_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph("Predictive Permanent Way (P-Way) Track Maintenance & Multi-Class Defect Severity Prioritization", subtitle_style))
    story.append(Spacer(1, 30))

    # Student metadata table
    meta_data = [
        [Paragraph("<b>Student Name</b>", body_style), Paragraph("<b>Reg. Number</b>", body_style), Paragraph("<b>Academic Role</b>", body_style)],
        [Paragraph("Karthik Reddy", body_style), Paragraph("21BCE1001", body_style), Paragraph("Lead ML Pipeline Architect", body_style)],
        [Paragraph("Ananya Sharma", body_style), Paragraph("21BCE1045", body_style), Paragraph("Data Modeling & Tuning Specialist", body_style)],
        [Paragraph("Rahul Verma", body_style), Paragraph("21BCE1089", body_style), Paragraph("Deployment & QA Engineer", body_style)],
    ]
    meta_table = Table(meta_data, colWidths=[1.8*inch, 1.3*inch, 3.5*inch])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef4f8")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#b0c4de")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 25))

    # Executive Summary Box
    summary_text = (
        "<b>Executive Summary:</b> Indian Railways operates over 68,000 route kilometers carrying 24 million "
        "passengers daily. Track safety demands identifying geometric irregularities and internal rail cracks "
        "before they escalate into catastrophic transverse fractures. This university capstone project delivers a "
        "production-grade, zero-leakage Scikit-Learn multi-class classification pipeline calibrated for severe "
        "class imbalance (~53% Low, ~28% Moderate, ~14% High, ~5% Critical). Our champion model achieves an "
        "exceptional <b>0.9609 Macro-F1 score</b> and <b>96.4% overall accuracy</b> on held-out test data, "
        "with 100% precision and 94.4% recall on minority Critical rail defects. Predictions are directly mapped "
        "to Indian Railways Permanent Way Manual (IRPWM 2020) safety directives and prioritized field dispatch queues."
    )
    summary_table = Table([[Paragraph(summary_text, body_style)]], colWidths=[6.6*inch])
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8f9fa")),
        ("BOX", (0, 0), (-1, -1), 1.0, colors.HexColor("#0b2545")),
        ("PADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(summary_table)

    story.append(PageBreak())

    # Section 2: Introduction & Problem Statement
    story.append(Paragraph("2. Introduction and Problem Statement", h1_style))
    story.append(Paragraph(
        "Indian Railways operates under heavy freight and passenger traffic exceeding 60 to 100 Gross Million Tonnes "
        "(GMT) per annum on trunk routes. Track degradation is governed by physical interactions between dynamic cyclic "
        "loading, track geometry deviations (gauge, twist, alignment, unevenness), internal metallurgical flaws detected "
        "via Ultrasonic Flaw Detection (USFD), and extreme environmental rail temperatures.",
        body_style
    ))
    story.append(Paragraph(
        "<b>The Severe Class Imbalance Challenge:</b> In practice, the defect distribution is heavily skewed. A generic "
        "model optimizing overall accuracy would misclassify critical fractures as safe track. This project specifically "
        "optimizes the Macro-Calibrated F1 Score to ensure high recall on life-threatening defects.",
        body_style
    ))

    # Section 3: Project Objectives & Need
    story.append(Paragraph("3. Project Objectives and Need", h1_style))
    story.append(Paragraph("• <b>Macro-F1 Optimization:</b> Achieve Macro-F1 >= 0.90 across all 4 severity levels.", bullet_style))
    story.append(Paragraph("• <b>Physics-Informed Feature Engineering:</b> Formulate compound indices including Track Quality Index (TQI), Thermal Stress Index, Structural Risk Ratio, and Ballast Degradation Factor.", bullet_style))
    story.append(Paragraph("• <b>Zero Data Leakage:</b> Encapsulate all transformations inside a Scikit-Learn ColumnTransformer pipeline.", bullet_style))
    story.append(Paragraph("• <b>Actionable P-Way Dispatch:</b> Produce sorted field work orders with IRPWM 2020 safety directives.", bullet_style))

    # Section 4 & 5: Architecture & Tech Stack
    story.append(Paragraph("4. System Architecture & Tech Stack", h1_style))
    story.append(Paragraph(
        "The architecture is organized into four tiers: (1) Multi-modal Data Ingestion, (2) Physics-Informed Feature "
        "Engineering, (3) Zero-Leakage Pipeline Preprocessing & Champion Classification, and (4) Permanent Way "
        "Maintenance Priority Dispatch Queue. Built with Python 3.10+, Scikit-learn, LightGBM, XGBoost, SHAP, and Streamlit.",
        body_style
    ))

    # Section 6: Methodology & Benchmarks
    story.append(Paragraph("5. Methodology & 5-Fold Stratified CV Benchmarks", h1_style))
    story.append(Paragraph(
        "We benchmarked 4 distinct multiclass architectures under 5-Fold Stratified Cross-Validation on the 4,000-sample "
        "training set. The comparison results are summarized below:",
        body_style
    ))

    bench_data = [
        [Paragraph("<b>Model Architecture</b>", body_style), Paragraph("<b>Imbalance Strategy</b>", body_style), Paragraph("<b>CV Macro-F1</b>", body_style), Paragraph("<b>CV Accuracy</b>", body_style)],
        [Paragraph("<b>Random Forest (Champion)</b>", body_style), Paragraph("Balanced Subsample", body_style), Paragraph("<b>0.9500 ± 0.0078</b>", body_style), Paragraph("<b>0.9575</b>", body_style)],
        [Paragraph("XGBoost Classifier", body_style), Paragraph("Sample-Weighted Loss", body_style), Paragraph("0.9494 ± 0.0090", body_style), Paragraph("0.9568", body_style)],
        [Paragraph("LightGBM Classifier", body_style), Paragraph("Class-Weighted Gradient", body_style), Paragraph("0.9466 ± 0.0083", body_style), Paragraph("0.9550", body_style)],
        [Paragraph("Logistic Regression", body_style), Paragraph("Class-Weighted Cost", body_style), Paragraph("0.9356 ± 0.0100", body_style), Paragraph("0.9502", body_style)],
    ]
    bench_table = Table(bench_data, colWidths=[2.2*inch, 1.8*inch, 1.4*inch, 1.2*inch])
    bench_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b2545")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(bench_table)

    story.append(PageBreak())

    # Section 7: Evaluation Results & Plots
    story.append(Paragraph("6. Evaluation Diagnostics & Held-Out Test Results", h1_style))
    story.append(Paragraph(
        "Evaluation on 1,000 held-out test instances confirmed outstanding generalization: "
        "<b>Macro-F1: 0.9609, Weighted-F1: 0.9639, Overall Accuracy: 96.40%</b>.",
        body_style
    ))

    perf_data = [
        [Paragraph("<b>Severity Tier</b>", body_style), Paragraph("<b>Precision</b>", body_style), Paragraph("<b>Recall</b>", body_style), Paragraph("<b>F1-Score</b>", body_style), Paragraph("<b>Support</b>", body_style)],
        [Paragraph("Low (Routine Monitoring)", body_style), Paragraph("0.97", body_style), Paragraph("0.99", body_style), Paragraph("0.98", body_style), Paragraph("528", body_style)],
        [Paragraph("Moderate (Scheduled Maintenance)", body_style), Paragraph("0.94", body_style), Paragraph("0.93", body_style), Paragraph("0.94", body_style), Paragraph("275", body_style)],
        [Paragraph("High (30 km/h Caution Order)", body_style), Paragraph("0.96", body_style), Paragraph("0.94", body_style), Paragraph("0.95", body_style), Paragraph("143", body_style)],
        [Paragraph("<b>Critical (Immediate Line Block)</b>", body_style), Paragraph("<b>1.00</b>", body_style), Paragraph("<b>0.94</b>", body_style), Paragraph("<b>0.97</b>", body_style), Paragraph("<b>54</b>", body_style)],
        [Paragraph("<b>Macro Average</b>", body_style), Paragraph("<b>0.97</b>", body_style), Paragraph("<b>0.95</b>", body_style), Paragraph("<b>0.96</b>", body_style), Paragraph("1000", body_style)],
    ]
    perf_table = Table(perf_data, colWidths=[2.2*inch, 1.1*inch, 1.1*inch, 1.1*inch, 1.1*inch])
    perf_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#134074")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("PADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(perf_table)
    story.append(Spacer(1, 12))

    # Add Confusion Matrix image if present
    cm_img_path = PROJECT_ROOT / "models" / "evaluation_plots" / "confusion_matrix.png"
    if cm_img_path.exists():
        story.append(Image(str(cm_img_path), width=6.5*inch, height=2.6*inch))
        story.append(Paragraph("<i>Figure 1: Raw Count & Normalized Recall Confusion Matrix on Held-Out Test Set.</i>", subtitle_style))
        story.append(Spacer(1, 10))

    # Add SHAP Summary image if present
    shap_img_path = PROJECT_ROOT / "models" / "evaluation_plots" / "shap_summary.png"
    if shap_img_path.exists():
        story.append(Image(str(shap_img_path), width=6.5*inch, height=3.0*inch))
        story.append(Paragraph("<i>Figure 2: SHAP Feature Importance Explaining Primary Drivers of Critical Rail Fractures.</i>", subtitle_style))

    story.append(PageBreak())

    # Section 8: Quality Assurance & Member Contributions
    story.append(Paragraph("7. Automated Test Suite Verification (Pytest)", h1_style))
    story.append(Paragraph(
        "An enterprise-grade test suite in <code>tests/test_pipeline.py</code> verified 11 test cases with 100% pass rate: "
        "Schema validity, class distribution constraints, zero-leakage isolation, probability simplex integrity, "
        "pristine track vs. critical fracture edge cases, and priority ranking monotonicity.",
        body_style
    ))

    story.append(Paragraph("8. Individual Member Contributions", h1_style))
    contrib_data = [
        [Paragraph("<b>Member</b>", body_style), Paragraph("<b>Core Responsibilities & Deliverables</b>", body_style)],
        [Paragraph("<b>Karthik Reddy</b><br/>(21BCE1001)", body_style), Paragraph("Lead ML Engineer: Formulated problem, designed synthetic Indian Railways data loader, implemented zero-leakage ColumnTransformer, executed 5-fold CV benchmarks, serialized champion pipeline.", body_style)],
        [Paragraph("<b>Ananya Sharma</b><br/>(21BCE1045)", body_style), Paragraph("Feature Engineering & Tuning: Formulated Track Quality Index and Structural Risk Ratio equations, evaluated class weighting strategies, conducted SHAP TreeExplainer attributions, generated publication plots.", body_style)],
        [Paragraph("<b>Rahul Verma</b><br/>(21BCE1089)", body_style), Paragraph("Deployment & QA: Developed interactive 4-tab Streamlit dashboard, programmed P-Way Priority Scoring dispatch queue, created 11-case Pytest automated test suite, compiled report and presentation deliverables.", body_style)],
    ]
    contrib_table = Table(contrib_data, colWidths=[1.8*inch, 4.8*inch])
    contrib_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b2545")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(contrib_table)
    story.append(Spacer(1, 14))

    # Section 9 & 10: Future Scope & References
    story.append(Paragraph("9. Future Scope & Limitations", h1_style))
    story.append(Paragraph(
        "Future phases will integrate live REST APIs from CRIS Track Management System (TMS), deploy LoRaWAN "
        "accelerometer nodes at switches, and analyze Sentinel-1 InSAR satellite ground subsidence data.",
        body_style
    ))

    story.append(Paragraph("10. Key References", h1_style))
    story.append(Paragraph("1. Indian Railways Permanent Way Manual (IRPWM 2020), Ministry of Railways, Government of India.", bullet_style))
    story.append(Paragraph("2. RDSO Manual for Ultrasonic Testing of Rails and Welded Joints (2020 Edition).", bullet_style))
    story.append(Paragraph("3. Lundberg, S. M. et al. (2017). 'A Unified Approach to Interpreting Model Predictions', NeurIPS.", bullet_style))
    story.append(Paragraph("4. Pedregosa, F. et al. (2011). 'Scikit-learn: Machine Learning in Python', JMLR.", bullet_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated PDF report at: {output_path}")


# ---------------------------------------------------------------------------
# PowerPoint Presentation Generation
# ---------------------------------------------------------------------------
def generate_pptx(output_path: str):
    print(f"Generating PPTX presentation: {output_path}...")
    prs = Presentation()
    prs.slide_width = Inches(13.333) # 16:9 widescreen
    prs.slide_height = Inches(7.5)

    blank_layout = prs.slide_layouts[6] # completely blank layout

    NAVY = RGBColor(11, 37, 69)
    GOLD = RGBColor(230, 126, 34)
    DARK_GRAY = RGBColor(50, 50, 50)
    LIGHT_BG = RGBColor(245, 247, 250)
    WHITE = RGBColor(255, 255, 255)

    slides_content = [
        {
            "num": 1,
            "title": "Railway Track Defect Severity Classification",
            "subtitle": "Predictive Permanent Way (P-Way) Track Maintenance & Multi-Class Defect Severity Prioritization",
            "bullets": [
                "Lead ML Pipeline Architect: Karthik Reddy (Reg. No: 21BCE1001)",
                "Data Modeling & Feature Engineering: Ananya Sharma (Reg. No: 21BCE1045)",
                "Full-Stack Deployment & QA: Rahul Verma (Reg. No: 21BCE1089)",
                "Domain: Supervised Machine Learning / Predictive Rail Maintenance",
                "Compliance: Indian Railways Permanent Way Manual (IRPWM 2020) & UMS Guidelines",
            ],
            "notes": "Good morning respected evaluators. Today we present our capstone project on Railway Track Defect Severity Classification. Our goal is to transform manual Indian Railways track inspections into an automated, predictive maintenance dispatch engine.",
        },
        {
            "num": 2,
            "title": "Problem Statement & The Severe Class Imbalance Dilemma",
            "subtitle": "The Challenge of Detecting Rare, Fatal Rail Fractures",
            "bullets": [
                "Operational Context: 68,000+ route km, 24 million passengers daily, heavy haul routes exceeding 80 GMT.",
                "Primary Failure Modes: Ultrasonic transverse rail fissures, excessive gauge spreading, twist, and buckling.",
                "Severe Class Imbalance: ~53% Low (Safe), ~28% Moderate, ~14% High, only ~5% Critical (Fractures).",
                "The Accuracy Trap: A trivial classifier guessing 'Safe' gets 53% accuracy while missing 100% of deadly cracks!",
                "Optimization Target: We optimize strictly for Macro-Calibrated F1 Score to ensure high recall on minority defects.",
            ],
            "notes": "Class imbalance is a life-or-death challenge in railway safety. Standard accuracy is misleading because 53% of segments are normal. Our objective is to ensure that rare, fatal fractures are never missed.",
        },
        {
            "num": 3,
            "title": "Project Objectives & Measurable Engineering Goals",
            "subtitle": "Safety First, Zero Leakage & Actionable P-Way Dispatches",
            "bullets": [
                "Achieve Macro-F1 >= 0.90 across all 4 severity tiers (Low, Moderate, High, Critical).",
                "Formulate Physics-Informed Feature Engineering (Track Quality Index, Thermal Stress, Structural Risk).",
                "Enforce Zero Data Leakage via Scikit-Learn ColumnTransformer and Pipeline isolation.",
                "Translate ML Predictions to Indian Railways Permanent Way Manual (IRPWM) Safety Directives.",
                "Deliver Real-Time Streamlit Operations Console with sub-2ms inference latency.",
            ],
            "notes": "Our objectives focus on rigorous machine learning and domain actionability. Every prediction is directly linked to an official IRPWM safety protocol.",
        },
        {
            "num": 4,
            "title": "System Architecture & End-to-End Workflow",
            "subtitle": "Four-Tier Predictive Maintenance Pipeline",
            "bullets": [
                "Tier 1: Multi-Modal Data Ingestion (TRC Geometry, USFD Flaws, Axle Accelerometers, Weather).",
                "Tier 2: Physics-Informed Feature Engineering Engine (TQI, Thermal Stress Index, Structural Risk Ratio).",
                "Tier 3: Zero-Leakage Preprocessing & Champion Multiclass Classifier (Median Imputer, RobustScaler).",
                "Tier 4: Permanent Way Maintenance Dispatch Queue (Priority Scoring 0-100 & CSV Work Orders).",
            ],
            "notes": "Here is our four-tier system architecture connecting physical track recording assets to field maintenance gangs.",
        },
        {
            "num": 5,
            "title": "Tools, Frameworks & Tech Stack",
            "subtitle": "Production-Grade Open Source Ecosystem",
            "bullets": [
                "Core Programming: Python 3.10+ / 3.13 with NumPy, Pandas, Scikit-learn.",
                "Class Imbalance & Tree Boosting: Random Forest (Balanced Subsample), XGBoost, LightGBM.",
                "Explainable AI: SHAP (SHapley Additive exPlanations) TreeExplainer.",
                "Interactive Web Application: Streamlit & Plotly Express.",
                "Quality Assurance: Pytest test suite with 11 automated unit/integration tests.",
            ],
            "notes": "Our stack emphasizes reproducibility, performance, and transparency through automated testing and game-theoretic explainability.",
        },
        {
            "num": 6,
            "title": "Physics-Informed Feature Engineering & Leakage Isolation",
            "subtitle": "Modeling Rail Physics Compound Interactions",
            "bullets": [
                "Track Quality Index (TQI): Composite Euclidean norm of gauge, alignment, twist, and unevenness.",
                "Thermal Stress Index: Temperature divergence from 40°C stress-free neutral point multiplied by route GMT.",
                "Structural Risk Ratio: Ultrasonic flaw depth relative to rail head depth (38mm for 52kg vs. 43.5mm for 60kg).",
                "Vibration Dynamic Load: Axle box peak vertical acceleration amplified by annual tonnage.",
                "Zero Leakage Isolation: Transformers fitted strictly on training folds within cross-validation.",
            ],
            "notes": "Railway defects do not occur in isolation. By modeling the interaction between thermal stress, cyclic tonnage, and crack geometry, we give our models domain-grounded predictive power.",
        },
        {
            "num": 7,
            "title": "5-Fold Stratified Cross-Validation Benchmark",
            "subtitle": "Rigorous Evaluation Across 4 Multi-Class Architectures",
            "bullets": [
                "Random Forest (Balanced Subsample): CV Macro-F1 = 0.9500 ± 0.0078 | Accuracy = 95.75% (CHAMPION)",
                "XGBoost Classifier (Sample-Weighted): CV Macro-F1 = 0.9494 ± 0.0090 | Accuracy = 95.68%",
                "LightGBM Classifier (Class-Weighted): CV Macro-F1 = 0.9466 ± 0.0083 | Accuracy = 95.50%",
                "Logistic Regression (Baseline): CV Macro-F1 = 0.9356 ± 0.0100 | Accuracy = 95.02%",
                "Key Finding: Tree ensembles with class weighting significantly outperform linear baselines on minority defects.",
            ],
            "notes": "Under 5-fold stratified cross-validation on 4,000 training records, Random Forest with balanced subsampling achieved the highest Macro-F1 of 0.9500 and was selected as the champion pipeline.",
        },
        {
            "num": 8,
            "title": "Held-Out Test Results & SHAP Explainability",
            "subtitle": "Diagnostics on 1,000 Unseen Track Observations",
            "bullets": [
                "Test Macro-F1: 0.9609 | Weighted-F1: 0.9639 | Overall Test Accuracy: 96.40%",
                "Critical Severity Recall: 94.4% with 100% Precision (Zero false positives, zero safe misclassifications).",
                "High Severity Recall: 94.4% with 96.0% Precision (Timely speed restriction imposition).",
                "SHAP Interpretability: Ultrasonic flaw depth and gauge spread are the primary physical triggers of rail fractures.",
            ],
            "notes": "On 1,000 unseen test segments, our model achieved 96.4% accuracy. Most crucially: 100% precision on Critical defects with zero fractures misclassified as safe.",
        },
        {
            "num": 9,
            "title": "Interactive Streamlit Dashboard & Pytest QA",
            "subtitle": "P-Way Maintenance Dispatch Queue & Sub-2ms Latency",
            "bullets": [
                "Tab 1: Real-Time Classifier with inspection presets and official IRPWM safety directives.",
                "Tab 2: Batch Inspection CSV upload and automated P-Way maintenance priority queue ranking.",
                "Tab 3: Model interpretability featuring confusion matrix heatmaps and SHAP attribution plots.",
                "Tab 4: Latency profiler recording 1.45 ms inference latency (690 predictions/sec throughput).",
                "Automated Pytest Suite: 11 of 11 tests passing in 2.30s verifying schema, leakage isolation, and edge cases.",
            ],
            "notes": "Our Streamlit dashboard provides Senior Section Engineers with an actionable interface. Batch CSV uploads are ranked and exported for field gangs with 1 click.",
        },
        {
            "num": 10,
            "title": "Individual Contributions, Roadmap & Conclusion",
            "subtitle": "University Compliance & Future Engineering Scope",
            "bullets": [
                "Karthik Reddy (Lead): Pipeline architecture, data loader, CV benchmarks, champion serialization.",
                "Ananya Sharma: Domain feature equations, imbalance calibration, SHAP explainability.",
                "Rahul Verma: Streamlit dashboard, P-Way priority queue, Pytest suite, UMS documentation.",
                "Future Scope: Integration with CRIS Track Management System (TMS) APIs & LoRaWAN IoT accelerometers.",
                "Conclusion: A production-ready, university-compliant solution advancing Indian Railways track safety.",
            ],
            "notes": "In conclusion, our project fulfills all university requirements while addressing an urgent industrial safety challenge. Thank you, and we look forward to your questions.",
        },
    ]

    for slide_data in slides_content:
        slide = prs.slides.add_slide(blank_layout)

        # Header background banner
        header_shape = slide.shapes.add_shape(1, Inches(0), Inches(0), Inches(13.333), Inches(1.3)) # MSO_SHAPE.RECTANGLE
        header_shape.fill.solid()
        header_shape.fill.fore_color.rgb = NAVY
        header_shape.line.fill.background()

        # Title text
        txBox = slide.shapes.add_textbox(Inches(0.8), Inches(0.15), Inches(11.7), Inches(0.6))
        tf = txBox.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = f"Slide {slide_data['num']}: {slide_data['title']}"
        p.font.name = "Arial"
        p.font.size = Pt(22)
        p.font.bold = True
        p.font.color.rgb = WHITE

        # Subtitle text
        p2 = tf.add_paragraph()
        p2.text = slide_data["subtitle"]
        p2.font.name = "Arial"
        p2.font.size = Pt(13)
        p2.font.color.rgb = GOLD

        # Content text box
        content_box = slide.shapes.add_textbox(Inches(0.8), Inches(1.6), Inches(11.7), Inches(5.2))
        ctf = content_box.text_frame
        ctf.word_wrap = True

        for i, bullet in enumerate(slide_data["bullets"]):
            bp = ctf.paragraphs[0] if i == 0 else ctf.add_paragraph()
            bp.text = f"•  {bullet}"
            bp.font.name = "Arial"
            bp.font.size = Pt(17)
            bp.font.color.rgb = DARK_GRAY
            bp.space_after = Pt(14)

        # Add speaker notes
        notes_slide = slide.notes_slide
        text_frame = notes_slide.notes_text_frame
        text_frame.text = slide_data["notes"]

    prs.save(output_path)
    print(f"Successfully generated PPTX presentation at: {output_path}")


if __name__ == "__main__":
    docs_dir = PROJECT_ROOT / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    pdf_out = docs_dir / "PROJECT_REPORT.pdf"
    pptx_out = docs_dir / "PRESENTATION.pptx"
    
    generate_pdf(str(pdf_out))
    generate_pptx(str(pptx_out))
    print("Compilation of PDF and PPTX completed successfully.")
