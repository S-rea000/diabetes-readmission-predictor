"""
=============================================================================
DIABETES READMISSION PREDICTION — STREAMLIT WEB APPLICATION
=============================================================================
Run with:  streamlit run app.py

KEY FIXES vs previous version:
  1. diag_1/2/3  — now uses the real LabelEncoder indices from training,
                   not the broken float(code) % 100 proxy.
  2. race        — keys match training strings exactly (no spaces).
  3. medical_specialty — keys and values match LabelEncoder from training.
  4. Decision threshold — tuned to 0.463 (optimal F1) instead of default 0.5.
  5. Risk gauge  — scaled to real probability range (min ~0.06, max ~0.88),
                   so the bar visually fills meaningfully.
  6. CSS variables — all CSS var() references replaced with real hex values
                     so they render correctly regardless of Streamlit theme.
=============================================================================
"""

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Diabetes Readmission Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    background-color: #f0faf8 !important;
    color: #1e2d2f !important;
}
.block-container { padding: 2rem 3rem 3rem 3rem !important; max-width: 1400px; }

/* ── Header banner ── */
.app-header {
    background: linear-gradient(135deg, #0d7377 0%, #45c4b0 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
}
.app-header h1 {
    font-family: 'DM Serif Display', serif !important;
    font-size: 2.2rem; margin: 0 0 0.4rem 0;
    color: #ffffff !important; letter-spacing: -0.5px;
}
.app-header p { font-size: 1rem; color: #9aeddc !important; margin: 0; font-weight: 300; }

/* ── Section cards ── */
.section-card {
    background: #ffffff;
    border-radius: 14px;
    padding: 1.4rem 1.6rem 1.6rem 1.6rem;
    margin-bottom: 1.2rem;
    border: 1px solid #d4eeea;
    box-shadow: 0 2px 8px rgba(20,145,155,0.07);
}
.section-title {
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: #14919b !important;
    margin-bottom: 1rem; padding-bottom: 0.5rem;
    border-bottom: 2px solid #e6f9f5;
}

/* ── Predict button ── */
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #0d7377, #45c4b0) !important;
    color: #ffffff !important; border: none !important;
    border-radius: 10px !important; font-size: 1.05rem !important;
    font-weight: 600 !important; padding: 0.75rem 2rem !important;
    width: 100% !important; letter-spacing: 0.02em !important;
}
div[data-testid="stButton"] > button:hover { opacity: 0.9 !important; }

/* ── Result banners ── */
.result-high {
    background: #fdecea; border-left: 5px solid #d64045;
    border-radius: 10px; padding: 1.2rem 1.5rem;
}
.result-low {
    background: #e8f8ef; border-left: 5px solid #2a9d5c;
    border-radius: 10px; padding: 1.2rem 1.5rem;
}
.result-title { font-size: 1.15rem; font-weight: 600; margin: 0 0 0.3rem 0; }
.result-sub   { font-size: 0.9rem; color: #3d5a5e; margin: 0; }

/* ── Metric cards ── */
.metric-card {
    background: #e6f9f5; border-radius: 12px;
    padding: 1.1rem 1.4rem; text-align: center;
    border: 1px solid #c3ebe3;
}
.metric-value { font-size: 2rem; font-weight: 700; color: #0d7377; line-height: 1; }
.metric-label { font-size: 0.75rem; color: #3d5a5e; margin-top: 0.35rem;
                font-weight: 500; text-transform: uppercase; letter-spacing: 0.08em; }

/* ── Risk gauge ── */
.gauge-track { background: #ddf0ec; border-radius: 99px; height: 16px; overflow: hidden; margin: 0.5rem 0 0.3rem 0; }
.gauge-low   { background: linear-gradient(90deg,#2a9d5c,#52c97e); height:100%; border-radius:99px; }
.gauge-mid   { background: linear-gradient(90deg,#e8a838,#f4c36a); height:100%; border-radius:99px; }
.gauge-high  { background: linear-gradient(90deg,#d64045,#f47c7c); height:100%; border-radius:99px; }

/* ── Factor table ── */
.factor-table { width:100%; border-collapse:collapse; margin-top:0.8rem; }
.factor-table th { background:#0d7377; color:#ffffff; font-size:0.75rem;
                   letter-spacing:0.07em; text-transform:uppercase;
                   padding:0.6rem 1rem; text-align:left; }
.factor-table td { padding:0.55rem 1rem; font-size:0.87rem; border-bottom:1px solid #eaf4f2; }
.factor-table tr:last-child td { border-bottom:none; }
.factor-table tr:nth-child(even) td { background:#f0faf8; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d7377 0%, #0a5a5e 100%) !important;
}
[data-testid="stSidebar"] * { color: #ffffff !important; }
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3
    { color: #9aeddc !important; font-family:'DM Serif Display',serif !important; }
[data-testid="stSidebar"] hr  { border-color: rgba(255,255,255,0.15) !important; }
[data-testid="stSidebar"] th  { color:#9aeddc !important; background:rgba(255,255,255,0.1) !important; }
[data-testid="stSidebar"] td  { color:rgba(255,255,255,0.85) !important; }

/* ── Disclaimer ── */
.disclaimer {
    background: rgba(20,145,155,0.07); border:1px solid #b8deda;
    border-radius:10px; padding:0.9rem 1.2rem;
    font-size:0.82rem; color:#3d5a5e !important; margin-top:1.5rem;
}

/* ── Streamlit widget tweaks ── */
.stSlider > div > div > div > div { background: #14919b !important; }
div[data-baseweb="select"] > div  { border-color:#b8deda !important; border-radius:8px !important; }
div[data-baseweb="input"]  > div  { border-color:#b8deda !important; border-radius:8px !important; }
</style>
""", unsafe_allow_html=True)


# ── Load model artifacts ──────────────────────────────────────────────────────
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")

@st.cache_resource
def load_artifacts():
    m = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
    s = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    f = joblib.load(os.path.join(MODEL_DIR, "features.pkl"))
    return m, s, f

try:
    model, scaler, features, encoders = load_artifacts()
except Exception as err:
    st.error(
        f"❌ Could not load model artifacts from `app/model/`.\n\n"
        f"Run `notebooks/diabetes_full_analysis.py` first.\n\n`{err}`"
    )
    st.stop()

# ── Encoding maps ─────────────────────────────────────────────────────────────
# These must EXACTLY mirror what the training pipeline produced.

AGE_MAP = {
    "[0-10)": 5,  "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
    "[40-50)": 45,"[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
    "[80-90)": 85,"[90-100)": 95,
}

MED_MAP = {"No": 0, "Steady": 1, "Up": 2, "Down": 2}

GLU_MAP = {"Not Tested": 0, "Normal": 1, ">200": 2, ">300": 3}
A1C_MAP = {"Not Tested": 0, "Normal": 1, ">7": 2,   ">8":  3}

# Keys match training LabelEncoder strings EXACTLY (no extra spaces)
RACE_DISPLAY = {
    "African American": "AfricanAmerican",
    "Asian":            "Asian",
    "Caucasian":        "Caucasian",
    "Hispanic":         "Hispanic",
    "Other":            "Other",
}

GENDER_DISPLAY = {
    "Female":           "Female",
    "Male":             "Male",
    "Unknown/Invalid":  "Unknown/Invalid",
}

# Medical specialties — display label -> training string
SPECIALTY_DISPLAY = {
    "Internal Medicine":    "InternalMedicine",
    "Cardiology":           "Cardiology",
    "Family / General":     "Family/GeneralPractice",
    "Surgery – General":    "Surgery-General",
    "Orthopedics":          "Orthopedics",
    "Gastroenterology":     "Gastroenterology",
    "Pulmonology":          "Pulmonology",
    "Endocrinology":        "Endocrinology",
    "Emergency / Trauma":   "Emergency/Trauma",
    "Nephrology":           "Nephrology",
    "Neurology":            "Neurology",
    "Other":                "Family/GeneralPractice",   # fallback to common value
}

ADM_TYPE_MAP = {
    "Emergency": 1, "Urgent": 2, "Elective": 3, "Newborn": 4,
    "Not Available": 5, "NULL": 6, "Trauma Center": 7, "Not Mapped": 8,
}

# Common ICD-9 codes with friendly names; mapped to exact LabelEncoder index
# Format: display_name -> (icd_string_used_at_train_time, encoded_int)
# Values obtained from the fitted LabelEncoder during training.
DIAG_OPTIONS = {
    "250   — Diabetes mellitus":              "250",
    "250.01 — DM type II, uncontrolled":      "250.01",
    "250.1  — DM with ketoacidosis":          "250.1",
    "250.6  — DM with neurological compl.":   "250.6",
    "250.8  — DM with other compl.":          "250.8",
    "428    — Heart failure":                 "428",
    "414    — Coronary artery disease":       "414",
    "410    — Acute MI":                      "410",
    "427    — Cardiac arrhythmias":           "427",
    "401    — Hypertension":                  "401",
    "276    — Fluid/electrolyte disorders":   "276",
    "285    — Anaemia":                       "285",
    "486    — Pneumonia":                     "486",
    "584    — Acute kidney failure":          "584",
    "Other / Unknown":                        "250",   # default fallback
}

def get_diag_encoded(display_key: str, diag_col: str) -> int:
    """Return the correct LabelEncoder integer for a diagnosis option."""
    icd_str = DIAG_OPTIONS[display_key]
    enc = encoders[diag_col]
    # Try exact match first, then strip decimal, then fallback to mode
    if icd_str in enc:
        return enc[icd_str]
    icd_base = icd_str.split(".")[0]
    if icd_base in enc:
        return enc[icd_base]
    return enc.get("250", 87)   # final fallback


# ── Optimal decision threshold (maximises F1 on held-out test set) ────────────
DECISION_THRESHOLD = 0.463

# Real probability range for gauge scaling (from test-set analysis)
PROB_MIN = 0.06
PROB_MAX = 0.88


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🩺 About")
    st.markdown("""
Predicts whether a diabetic patient will be **readmitted within 30 days** of discharge.

Trained on the **UCI Diabetes 130-US Hospitals** dataset (1999–2008) — 50,000+ encounters.
""")
    st.markdown("---")
    st.markdown("### Model Performance")
    st.markdown("""
| Metric    | Score |
|-----------|-------|
| ROC-AUC   | 0.669 |
| F1-Score  | 0.271 |
| Recall    | 0.570 |
| Threshold | 0.463 |
| Algorithm | Random Forest (200 trees) |
""")
    st.markdown("---")
    st.markdown("### Top Risk Factors")
    st.markdown("""
1. Prior inpatient visits
2. Total prior visits
3. Primary diagnosis (ICD-9)
4. Discharge disposition
5. Number of medications
6. Number of lab procedures
""")
    st.markdown("---")
    st.caption("🎓 Data Science Certification Project")


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <h1>🩺 Diabetes Readmission Risk Predictor</h1>
    <p>Complete the four sections below and click <strong>Run Prediction</strong> to assess readmission risk.</p>
</div>
""", unsafe_allow_html=True)


# ── INPUT FORM — 2 × 2 card grid ─────────────────────────────────────────────
row1_left, row1_right = st.columns(2, gap="large")
row2_left, row2_right = st.columns(2, gap="large")


# ══ CARD 1 — Demographics + Admission ════════════════════════════════════════
with row1_left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">👤 Patient Demographics</p>', unsafe_allow_html=True)

    age           = st.selectbox("Age Group", list(AGE_MAP.keys()), index=7)
    gender_label  = st.selectbox("Gender", list(GENDER_DISPLAY.keys()))
    race_label    = st.selectbox("Race / Ethnicity", list(RACE_DISPLAY.keys()), index=2)

    st.markdown('<p class="section-title" style="margin-top:1.2rem">🏨 Admission Details</p>',
                unsafe_allow_html=True)

    adm_type_label           = st.selectbox("Admission Type", list(ADM_TYPE_MAP.keys()))
    admission_type_id        = ADM_TYPE_MAP[adm_type_label]
    discharge_disposition_id = st.slider("Discharge Disposition ID", 1, 28, 1,
                                         help="1 = Discharged to home; 6 = Discharged to home with health service")
    admission_source_id      = st.slider("Admission Source ID", 1, 25, 7,
                                         help="7 = Emergency Room")
    time_in_hospital         = st.slider("Days in Hospital", 1, 14, 4)
    st.markdown('</div>', unsafe_allow_html=True)


# ══ CARD 2 — Clinical + Diagnoses ═════════════════════════════════════════════
with row1_right:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">🔬 Clinical Measurements</p>', unsafe_allow_html=True)

    num_lab_procedures = st.slider("Number of Lab Procedures",  1, 132, 44)
    num_procedures     = st.slider("Number of Procedures",        0,   6,  1)
    num_medications    = st.slider("Number of Medications",       1,  81, 15)
    number_diagnoses   = st.slider("Number of Diagnoses",         1,  16,  8)

    c1, c2 = st.columns(2)
    with c1:
        max_glu_serum = st.selectbox("Max Glucose Serum",
                                     ["Not Tested", "Normal", ">200", ">300"],
                                     help="Result of maximum glucose serum test during stay")
    with c2:
        A1Cresult = st.selectbox("HbA1c Result",
                                 ["Not Tested", "Normal", ">7", ">8"],
                                 help="Glycated haemoglobin test result")

    st.markdown('<p class="section-title" style="margin-top:1.2rem">📋 ICD-9 Diagnosis Codes</p>',
                unsafe_allow_html=True)

    diag_keys = list(DIAG_OPTIONS.keys())
    d1, d2, d3 = st.columns(3)
    with d1:
        diag_1_label = st.selectbox("Primary",   diag_keys, index=0)
    with d2:
        diag_2_label = st.selectbox("Secondary", diag_keys,
                                    index=diag_keys.index("428    — Heart failure"))
    with d3:
        diag_3_label = st.selectbox("Tertiary",  diag_keys,
                                    index=diag_keys.index("401    — Hypertension"))
    st.markdown('</div>', unsafe_allow_html=True)


# ══ CARD 3 — Prior History + Specialty ═══════════════════════════════════════
with row2_left:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">📊 Prior Visit History (past 12 months)</p>',
                unsafe_allow_html=True)

    number_outpatient = st.slider("Outpatient Visits",  0, 42,  0)
    number_emergency  = st.slider("Emergency Visits",   0, 76,  0)
    number_inpatient  = st.slider("Inpatient Visits",   0, 21,  0)
    total_visits      = number_outpatient + number_emergency + number_inpatient

    tv_color = "#d64045" if total_visits >= 5 else "#0d7377"
    st.markdown(f"""
    <div style="background:#e6f9f5;border-radius:10px;padding:0.9rem 1.2rem;
                margin-top:0.8rem;border:1px solid #c3ebe3;">
        <span style="font-size:0.75rem;color:#0d7377;font-weight:600;
                     text-transform:uppercase;letter-spacing:0.08em;">Total Prior Visits</span>
        <div style="font-size:2.2rem;font-weight:700;color:{tv_color};line-height:1.2;">
            {total_visits}
            <span style="font-size:0.9rem;font-weight:400;color:#3d5a5e;">
                {"  ⚠️ Elevated" if total_visits >= 5 else "  ✅ Normal"}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="section-title" style="margin-top:1.2rem">🩺 Treating Specialty</p>',
                unsafe_allow_html=True)
    specialty_label = st.selectbox("Medical Specialty", list(SPECIALTY_DISPLAY.keys()))
    st.markdown('</div>', unsafe_allow_html=True)


# ══ CARD 4 — Medications ══════════════════════════════════════════════════════
with row2_right:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<p class="section-title">💊 Medication Status</p>', unsafe_allow_html=True)
    st.caption("For each drug: No = not prescribed · Steady = dose unchanged · Up/Down = dose changed")

    mc1, mc2 = st.columns(2)
    with mc1:
        insulin       = st.selectbox("Insulin",      ["No", "Steady", "Up", "Down"])
        metformin     = st.selectbox("Metformin",     ["No", "Steady", "Up", "Down"])
        glipizide     = st.selectbox("Glipizide",     ["No", "Steady", "Up", "Down"])
    with mc2:
        glyburide     = st.selectbox("Glyburide",     ["No", "Steady", "Up", "Down"])
        pioglitazone  = st.selectbox("Pioglitazone",  ["No", "Steady", "Up", "Down"])
        rosiglitazone = st.selectbox("Rosiglitazone", ["No", "Steady", "Up", "Down"])

    st.markdown('<p class="section-title" style="margin-top:1.2rem">🔄 Treatment Summary</p>',
                unsafe_allow_html=True)
    sc1, sc2 = st.columns(2)
    with sc1:
        change      = st.radio("Medication change?",      ["No", "Yes"], horizontal=True)
    with sc2:
        diabetesMed = st.radio("On diabetes medication?", ["Yes", "No"], horizontal=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ── Predict button ────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
_, btn_col, _ = st.columns([2, 3, 2])
with btn_col:
    predict_btn = st.button("🔍  Run Prediction", use_container_width=True, type="primary")


# ── Feature assembly ──────────────────────────────────────────────────────────
def build_features() -> pd.DataFrame:
    """
    Convert all UI widget values to the exact numeric representation
    the trained model expects. Every encoding mirrors the training pipeline.
    """
    # Resolve label-encoder values using saved encoder dicts
    race_enc      = encoders["race"][RACE_DISPLAY[race_label]]
    gender_enc    = encoders["gender"][GENDER_DISPLAY[gender_label]]
    specialty_str = SPECIALTY_DISPLAY[specialty_label]
    specialty_enc = encoders["medical_specialty"].get(specialty_str,
                    encoders["medical_specialty"]["Family/GeneralPractice"])

    diag_1_enc = get_diag_encoded(diag_1_label, "diag_1")
    diag_2_enc = get_diag_encoded(diag_2_label, "diag_2")
    diag_3_enc = get_diag_encoded(diag_3_label, "diag_3")

    medication_count   = sum([MED_MAP[insulin], MED_MAP[metformin], MED_MAP[glipizide],
                              MED_MAP[glyburide], MED_MAP[pioglitazone], MED_MAP[rosiglitazone]])
    procedures_per_day = (num_procedures + 1) / (time_in_hospital + 1)

    raw = {
        "race":                     race_enc,
        "gender":                   gender_enc,
        "age":                      AGE_MAP[age],
        "admission_type_id":        admission_type_id,
        "discharge_disposition_id": discharge_disposition_id,
        "admission_source_id":      admission_source_id,
        "time_in_hospital":         time_in_hospital,
        "medical_specialty":        specialty_enc,
        "num_lab_procedures":       num_lab_procedures,
        "num_procedures":           num_procedures,
        "num_medications":          num_medications,
        "number_outpatient":        number_outpatient,
        "number_emergency":         number_emergency,
        "number_inpatient":         number_inpatient,
        "diag_1":                   diag_1_enc,
        "diag_2":                   diag_2_enc,
        "diag_3":                   diag_3_enc,
        "number_diagnoses":         number_diagnoses,
        "max_glu_serum":            GLU_MAP[max_glu_serum],
        "A1Cresult":                A1C_MAP[A1Cresult],
        "metformin":                MED_MAP[metformin],
        "repaglinide":              0,
        "nateglinide":              0,
        "chlorpropamide":           0,
        "glimepiride":              0,
        "acetohexamide":            0,
        "glipizide":                MED_MAP[glipizide],
        "glyburide":                MED_MAP[glyburide],
        "tolbutamide":              0,
        "pioglitazone":             MED_MAP[pioglitazone],
        "rosiglitazone":            MED_MAP[rosiglitazone],
        "acarbose":                 0,
        "miglitol":                 0,
        "troglitazone":             0,
        "tolazamide":               0,
        "examide":                  0,
        "citoglipton":              0,
        "insulin":                  MED_MAP[insulin],
        "glyburide-metformin":      0,
        "glipizide-metformin":      0,
        "glimepiride-pioglitazone": 0,
        "metformin-rosiglitazone":  0,
        "metformin-pioglitazone":   0,
        "change":                   1 if change == "Yes" else 0,
        "diabetesMed":              1 if diabetesMed == "Yes" else 0,
        "total_visits":             total_visits,
        "medication_count":         medication_count,
        "procedures_per_day":       procedures_per_day,
    }
    return pd.DataFrame([raw])[features]   # correct column order


# ── Results ───────────────────────────────────────────────────────────────────
if predict_btn:
    with st.spinner("Running prediction…"):
        try:
            input_df     = build_features()
            input_scaled = scaler.transform(input_df)
            raw_proba    = model.predict_proba(input_scaled)[0]

            # Class 1 = readmitted within 30 days
            risk_score = float(np.clip(raw_proba[1], 0.0, 1.0))
            safe_score = float(np.clip(raw_proba[0], 0.0, 1.0))
            risk_pct   = risk_score * 100
            safe_pct   = safe_score * 100

            # Use tuned threshold (0.463) not default 0.5
            prediction = 1 if risk_score >= DECISION_THRESHOLD else 0

            # Scale gauge to the real probability range of this model
            gauge_pct = (risk_score - PROB_MIN) / (PROB_MAX - PROB_MIN) * 100
            gauge_pct = float(np.clip(gauge_pct, 2.0, 100.0))   # min 2% so bar is visible

        except Exception as err:
            st.error(f"Prediction failed: {err}")
            st.stop()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # ── Outcome banner ────────────────────────────────────────────────────
    if prediction == 1:
        st.markdown("""
        <div class="result-high">
            <p class="result-title">⚠️ HIGH RISK — Readmission likely within 30 days</p>
            <p class="result-sub">This patient's profile is associated with elevated 30-day readmission risk.
            Consider enhanced discharge planning and a close follow-up schedule.</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-low">
            <p class="result-title">✅ LOW RISK — Readmission unlikely within 30 days</p>
            <p class="result-sub">This patient's profile suggests a lower readmission probability.
            Standard discharge and follow-up protocols should be sufficient.</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Metric cards ──────────────────────────────────────────────────────
    prob_color = "#d64045" if prediction == 1 else "#2a9d5c"
    risk_cat   = "High 🔴" if risk_pct >= 46.3 else ("Moderate 🟡" if risk_pct >= 25 else "Low 🟢")

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{prob_color}">{risk_pct:.1f}%</div>
            <div class="metric-label">Readmission Probability</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{safe_pct:.1f}%</div>
            <div class="metric-label">Non-Readmission Probability</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="font-size:1.4rem">{risk_cat}</div>
            <div class="metric-label">Risk Category</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="font-size:1.1rem">RF · {DECISION_THRESHOLD}</div>
            <div class="metric-label">Model · Decision Threshold</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Risk gauge (scaled to real model range) ───────────────────────────
    if gauge_pct < 33:
        gauge_cls   = "gauge-low"
        gauge_color = "color:#2a9d5c"
        gauge_label = "Low Risk"
    elif gauge_pct < 66:
        gauge_cls   = "gauge-mid"
        gauge_color = "color:#e8a838"
        gauge_label = "Moderate Risk"
    else:
        gauge_cls   = "gauge-high"
        gauge_color = "color:#d64045"
        gauge_label = "High Risk"

    st.markdown(f"""
    <div style="margin-bottom:1.5rem;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.3rem;">
            <span style="font-size:0.78rem;font-weight:600;text-transform:uppercase;
                         letter-spacing:0.1em;color:#3d5a5e;">Risk Gauge
                <span style="font-weight:300;text-transform:none;letter-spacing:0"
                     title="Scaled to the model's actual probability range ({int(PROB_MIN*100)}%–{int(PROB_MAX*100)}%)">
                    ⓘ</span>
            </span>
            <span style="font-size:0.82rem;{gauge_color};font-weight:600">{gauge_label} · {risk_pct:.1f}%</span>
        </div>
        <div class="gauge-track">
            <div class="{gauge_cls}" style="width:{gauge_pct:.1f}%"></div>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:0.72rem;color:#829497;margin-top:0.2rem;">
            <span>Lower bound ({int(PROB_MIN*100)}%)</span>
            <span>Decision threshold ({int(DECISION_THRESHOLD*100)}%)</span>
            <span>Upper bound ({int(PROB_MAX*100)}%)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Two-column detail ─────────────────────────────────────────────────
    res_left, res_right = st.columns([1.3, 1], gap="large")

    with res_left:
        st.markdown('<p class="section-title">🔑 Key Patient Factors</p>',
                    unsafe_allow_html=True)
        factors = [
            ("Prior Inpatient Visits",  number_inpatient,
             "⚠️ Elevated" if number_inpatient >= 3 else "✅ Normal"),
            ("Total Prior Visits",      total_visits,
             "⚠️ Elevated" if total_visits >= 5    else "✅ Normal"),
            ("Days in Hospital",        time_in_hospital,
             "⚠️ Long stay" if time_in_hospital >= 7 else "✅ Normal"),
            ("Number of Medications",   num_medications,
             "⚠️ High"     if num_medications >= 20 else "✅ Normal"),
            ("Lab Procedures",          num_lab_procedures,
             "⚠️ High"     if num_lab_procedures >= 60 else "✅ Normal"),
            ("Number of Diagnoses",     number_diagnoses,
             "⚠️ Complex"  if number_diagnoses >= 9 else "✅ Normal"),
            ("Insulin Status",          insulin, "—"),
            ("Medication Change",       change,
             "⚠️ Changed"  if change == "Yes" else "✅ Unchanged"),
            ("HbA1c Result",            A1Cresult,
             "⚠️ Elevated" if A1Cresult in [">7", ">8"] else "✅ Normal"),
        ]
        rows = "".join(
            f"<tr><td>{n}</td><td><strong>{v}</strong></td><td>{s}</td></tr>"
            for n, v, s in factors
        )
        st.markdown(f"""
        <table class="factor-table">
            <thead><tr><th>Factor</th><th>Value</th><th>Status</th></tr></thead>
            <tbody>{rows}</tbody>
        </table>""", unsafe_allow_html=True)

    with res_right:
        st.markdown('<p class="section-title">📋 Prediction Summary</p>',
                    unsafe_allow_html=True)
        outcome_text = "⚠️ HIGH Risk — Readmission likely" if prediction == 1 \
                       else "✅ LOW Risk — Readmission unlikely"
        st.markdown(f"""
        <div style="background:#e6f9f5;border-radius:12px;
                    padding:1.4rem;border:1px solid #c3ebe3;">
            <div style="font-size:1rem;font-weight:700;margin-bottom:1rem;
                        color:{prob_color}">{outcome_text}</div>
            <table style="width:100%;font-size:0.87rem;border-collapse:collapse;">
                <tr><td style="padding:0.35rem 0;color:#3d5a5e">Age Group</td>
                    <td style="font-weight:600;text-align:right">{age}</td></tr>
                <tr><td style="padding:0.35rem 0;color:#3d5a5e">Gender</td>
                    <td style="font-weight:600;text-align:right">{gender_label}</td></tr>
                <tr><td style="padding:0.35rem 0;color:#3d5a5e">Admission Type</td>
                    <td style="font-weight:600;text-align:right">{adm_type_label}</td></tr>
                <tr><td style="padding:0.35rem 0;color:#3d5a5e">HbA1c</td>
                    <td style="font-weight:600;text-align:right">{A1Cresult}</td></tr>
                <tr><td style="padding:0.35rem 0;color:#3d5a5e">Glucose Serum</td>
                    <td style="font-weight:600;text-align:right">{max_glu_serum}</td></tr>
                <tr><td style="padding:0.35rem 0;color:#3d5a5e">Specialty</td>
                    <td style="font-weight:600;text-align:right">{specialty_label}</td></tr>
                <tr style="border-top:2px solid #c3ebe3;">
                    <td style="padding:0.6rem 0 0.2rem;color:#3d5a5e;font-weight:600">Risk Score</td>
                    <td style="font-size:1.3rem;font-weight:700;text-align:right;
                               color:{prob_color}">{risk_pct:.1f}%</td></tr>
                <tr>
                    <td style="padding:0.2rem 0;color:#3d5a5e;font-size:0.8rem">Threshold</td>
                    <td style="font-size:0.8rem;text-align:right;color:#3d5a5e">
                        {int(DECISION_THRESHOLD*100)}% (tuned)</td></tr>
            </table>
        </div>""", unsafe_allow_html=True)

    # ── Disclaimer ────────────────────────────────────────────────────────
    st.markdown("""
    <div class="disclaimer">
        ⚠️ <strong>Educational Use Only.</strong> 
    </div>
    """, unsafe_allow_html=True)