import streamlit as st
import pandas as pd
import numpy as np
import joblib, os
from pathlib import Path
from project_paths import ORIGINAL_MODELS
from churn_features import build_admission_features, rebuild_features
from churn_pipeline import PipelineStore, legacy_test_data
from training_ui import render_training_page, render_batch_prediction

def lazy_import_sklearn():
    global train_test_split, accuracy_score, f1_score, roc_auc_score
    global confusion_matrix, classification_report, roc_curve, precision_score, recall_score
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                                 confusion_matrix, classification_report,
                                 roc_curve, precision_score, recall_score)

import plotly.express as px
import plotly.graph_objects as go

FOLDER = str(ORIGINAL_MODELS)

st.set_page_config(page_title="Interpretable Student Churn Prediction Using Machine Learning with Adaptive Semester Weighting", layout="wide",
                   initial_sidebar_state="expanded")

GOLD = "#C8A96E"
NAVY = "#1C2B3A"
GREEN = "#2D6A4F"
RED = "#8B2525"
AMBER = "#9B6A1A"
CREAM = "#FAFAF7"

def rgba(h, a=0.10):
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{a})"

# ── CSS ────────────────────────────────────────────────────
st.markdown(f"""<style>
*,*::before,*::after{{font-family:"Times New Roman",Times,serif!important;box-sizing:border-box;}}
.stApp{{background-color:{CREAM};}}
[data-testid="stSidebar"]{{background-color:{NAVY};border-right:4px solid {GOLD};}}
[data-testid="stSidebar"] *{{color:#EDE8DC!important;font-family:"Times New Roman",Times,serif!important;}}
[data-testid="stSidebar"] .stRadio label{{font-size:14px!important;padding:5px 0;display:block;}}

/* ══════════════════════════════════════════════════
   SIDEBAR ARROW FIX — font-size:0 kills text nodes,
   display:none kills child elements,
   ::after injects clean Unicode arrow
   ══════════════════════════════════════════════════ */

/* Collapsed button — FINAL FIX */
[data-testid="collapsedControl"] {{
    width:2rem!important;
    height:2rem!important;
    overflow:hidden!important;
}}
[data-testid="collapsedControl"] button {{
    font-size:0px!important;
    color:transparent!important;
    background:transparent!important;
    border:none!important;
    cursor:pointer!important;
    position:relative!important;
    width:100%!important;
    height:100%!important;
    display:flex!important;
    align-items:center!important;
    justify-content:center!important;
    overflow:visible!important;
    padding:0!important;
    margin:0!important;
    text-indent:-9999px!important;
    line-height:0!important;
    letter-spacing:-9999px!important;
}}
[data-testid="collapsedControl"] button *,
[data-testid="collapsedControl"] svg,
[data-testid="collapsedControl"] svg * {{
    display:none!important;
    visibility:hidden!important;
    width:0px!important;
    height:0px!important;
    font-size:0px!important;
}}
[data-testid="collapsedControl"] button::after {{
    content:"❯"!important;
    display:block!important;
    font-size:22px!important;
    font-weight:900!important;
    color:{NAVY}!important;
    visibility:visible!important;
    position:absolute!important;
    top:50%!important;
    left:50%!important;
    transform:translate(-50%,-50%)!important;
    width:auto!important;
    height:auto!important;
    line-height:1!important;
    font-family:Arial,sans-serif!important;
    text-indent:0px!important;
    letter-spacing:0px!important;
}}

/* Open sidebar button (double left arrow — sidebar visible) */
[data-testid="stSidebarCollapseButton"] {{
    width:2rem!important;
    height:2rem!important;
}}
[data-testid="stSidebarCollapseButton"] button {{
    font-size:0px!important;
    color:transparent!important;
    background:transparent!important;
    border:none!important;
    cursor:pointer!important;
    position:relative!important;
    width:100%!important;
    height:100%!important;
    display:flex!important;
    align-items:center!important;
    justify-content:center!important;
    overflow:visible!important;
    padding:0!important;
    margin:0!important;
}}
[data-testid="stSidebarCollapseButton"] svg {{
    display:none!important;
    visibility:hidden!important;
    width:0px!important;
    height:0px!important;
}}
[data-testid="stSidebarCollapseButton"] button * {{
    display:none!important;
    font-size:0px!important;
    visibility:hidden!important;
    width:0px!important;
    height:0px!important;
}}
[data-testid="stSidebarCollapseButton"] button::after {{
    content:"«"!important;
    display:flex!important;
    font-size:28px!important;
    font-weight:900!important;
    color:{GOLD}!important;
    visibility:visible!important;
    position:absolute!important;
    top:50%!important;
    left:50%!important;
    transform:translate(-50%,-50%)!important;
    width:auto!important;
    height:auto!important;
    line-height:1!important;
    font-family:Arial,sans-serif!important;
    letter-spacing:-5px!important;
    padding:0!important;
    margin:0!important;
}}

h1{{font-size:2.1rem!important;font-weight:700!important;color:{NAVY}!important;border-bottom:3px solid {GOLD};padding-bottom:9px;margin-bottom:8px;}}
h2{{font-size:1.4rem!important;font-weight:700!important;color:{NAVY}!important;margin-top:16px!important;}}
h3{{font-size:1.05rem!important;font-weight:600!important;color:#2D4A5E!important;}}
[data-testid="metric-container"]{{background:#FFF;border:1px solid #D4C5A9;border-top:4px solid {GOLD};border-radius:0;padding:14px 16px;box-shadow:2px 2px 7px rgba(0,0,0,0.06);}}
[data-testid="stMetricValue"]{{color:{NAVY}!important;font-size:1.85rem!important;font-weight:700!important;}}
[data-testid="stMetricLabel"]{{color:#5C6B7A!important;font-size:0.76rem!important;text-transform:uppercase;letter-spacing:1.2px;}}
.stTabs [data-baseweb="tab-list"]{{background:{NAVY};border-radius:0;gap:0;padding:0;}}
.stTabs [data-baseweb="tab"]{{background:transparent;color:{GOLD}!important;font-size:12.5px!important;font-weight:600;padding:10px 18px;border-right:1px solid rgba(200,169,110,0.2);border-radius:0;transition:all 0.2s ease;}}
.stTabs [data-baseweb="tab"]:hover{{background:rgba(200,169,110,0.15);}}
.stTabs [aria-selected="true"]{{background:{GOLD}!important;color:{NAVY}!important;}}
.stButton>button,[data-testid="stFormSubmitButton"]>button{{background-color:{NAVY};color:{GOLD};border:2px solid {GOLD};border-radius:0;font-size:13.5px;font-weight:700;padding:11px 26px;letter-spacing:1.2px;text-transform:uppercase;width:100%;transition:all 0.3s ease;}}
.stButton>button:hover,[data-testid="stFormSubmitButton"]>button:hover{{background-color:{GOLD};color:{NAVY};transform:translateY(-2px);box-shadow:0 4px 8px rgba(0,0,0,0.15);}}
.stSelectbox>label,.stNumberInput>label{{font-weight:700!important;color:{NAVY}!important;font-size:11.5px!important;text-transform:uppercase;letter-spacing:0.6px;}}
.stSelectbox>div>div,.stNumberInput>div>div{{border:2px solid #D4C5A9!important;border-radius:0!important;}}
.stDataFrame{{border:1px solid #D4C5A9;box-shadow:2px 2px 5px rgba(0,0,0,0.04);}}
hr{{border-color:#D4C5A9!important;margin:16px 0;}}
.stPlotlyChart{{border:1px solid #D4C5A9;background:#FFF;padding:10px;box-shadow:2px 2px 6px rgba(0,0,0,0.05);margin-bottom:12px;}}
details summary{{cursor:pointer;padding:10px 14px;background:{NAVY};color:{GOLD};font-size:12px;font-weight:700;letter-spacing:0.8px;text-transform:uppercase;list-style:none;transition:all 0.2s ease;}}
details summary:hover{{background:rgba(28,43,58,0.9);}}
details summary::-webkit-details-marker{{display:none;}}
details[open] summary{{border-bottom:2px solid {GOLD};}}
/* Replace only the native expander arrow; keep existing text and theme. */
[data-testid="stExpander"] summary [data-testid="stIconMaterial"]{{font-size:0!important;letter-spacing:0!important;transform:none!important;flex:0 0 1.25rem;width:1.25rem;}}
[data-testid="stExpander"] summary [data-testid="stIconMaterial"]::before{{content:"▸";font-family:Arial,sans-serif!important;font-size:18px;line-height:1;}}
[data-testid="stExpander"] details[open] summary [data-testid="stIconMaterial"]::before{{content:"▾";}}
details .insight-body{{background:#FFF;border:1px solid #D4C5A9;border-top:none;padding:14px 16px;font-size:13px;color:#2C3E50;line-height:1.85;}}
code{{background:#F5F2EB;padding:2px 6px;color:{NAVY};font-size:12px;border-radius:2px;border:1px solid #E5DCC8;}}
[data-testid="stForm"]{{border:2px solid {GOLD}!important;background:#FFF!important;padding:16px!important;}}
</style>""", unsafe_allow_html=True)

# ── HELPERS ────────────────────────────────────────────────
def section_header(title, sub=""):
    s = f"<div style='font-size:12px;color:#5C6B7A;margin-top:4px'>{sub}</div>" if sub else ""
    st.markdown(f"""<div style='border-left:5px solid {GOLD};padding:9px 17px;background:#FFF;
         margin-bottom:17px;box-shadow:2px 2px 5px rgba(0,0,0,0.05)'>
      <div style='font-size:18px;font-weight:700;color:{NAVY}'>{title}</div>{s}</div>""",
        unsafe_allow_html=True)

def info_card(html, color=NAVY):
    st.markdown(f"""<div style='background:#FFF;border:1px solid #D4C5A9;border-left:5px solid {color};
         padding:12px 16px;margin-bottom:10px;font-size:13px;color:#2C3E50;line-height:1.75'>{html}</div>""",
        unsafe_allow_html=True)

def section_banner(text):
    st.markdown(f"""<div style='background:{NAVY};color:{GOLD};font-size:14px;font-weight:700;
         letter-spacing:1.3px;text-transform:uppercase;padding:10px 20px;margin:22px 0 16px 0'>
      &#9672;&nbsp; {text}</div>""", unsafe_allow_html=True)

def tnr(fig, h=360):
    fig.update_layout(height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#FEFEFE",
        font=dict(family="Times New Roman,Times,serif", color=NAVY, size=12),
        title_font=dict(family="Times New Roman,Times,serif", color=NAVY, size=14),
        title_x=0.5,
        xaxis=dict(gridcolor="#EAE3D5", linecolor=GOLD, tickfont=dict(family="Times New Roman",size=11)),
        yaxis=dict(gridcolor="#EAE3D5", linecolor=GOLD, tickfont=dict(family="Times New Roman",size=11)),
        legend=dict(bgcolor="rgba(255,255,255,0.95)", bordercolor=GOLD, borderwidth=1,
                    font=dict(family="Times New Roman",size=11)),
        margin=dict(t=60, b=50, l=60, r=100))
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    return fig

def get_risk(p):
    if p < 0.20: return "Low Risk",       "#2D6A4F"
    elif p < 0.40: return "Moderate Risk",  "#9B6A1A"
    elif p < 0.65: return "High Risk",      "#8B2525"
    else:          return "Very High Risk", "#5C0A0A"

def chart_insight(title, body):
    st.markdown(f"""<details>
      <summary>&#9432;&nbsp; {title}</summary>
      <div class="insight-body">{body}</div>
    </details>""", unsafe_allow_html=True)

def form_label(text):
    st.markdown(f"""<div style='font-size:11px;font-weight:700;color:{GOLD};
         text-transform:uppercase;letter-spacing:0.8px;
         border-bottom:1px solid rgba(200,169,110,0.5);
         padding-bottom:4px;margin:14px 0 10px 0'>{text}</div>""", unsafe_allow_html=True)

# ── CONSTANTS ──────────────────────────────────────────────
HIGH_CHURN_DISTS = ['SABARKANTHA','KUTCH','BANASKANTHA','MEHSANA','BOTAD']
LOCAL_DISTS      = ['GANDHINAGAR','AHMEDABAD']
ALL_DISTRICTS    = sorted([
    "AHMEDABAD","AMRELI","ARVALLI","BANASKANTHA","BHAVNAGAR","BOTAD",
    "GANDHINAGAR","GIR SOMNATH","JAMNAGAR","JUNAGADH","KHEDA","KUTCH",
    "MEHSANA","MORBI","PATAN","RAJKOT","SABARKANTHA","SURENDRANAGAR",
    "ANAND","AURANGABAD","DAHOD","DEVBHUMI DWARKA","DUNGARPUR","GONDIYA",
    "HALVAD","HIMMATNAGAR","MAHARASHTRA","MAHISAGAR","NAVSARI","PORBANDAR",
    "SIROHI","SURAT",
])

SEM_HIST_RATE = {1: 1.000, 2: 0.809, 3: 0.032, 4: 0.000, 5: 0.000, 6: 0.000}

COLLEGE_SEMESTERS = {
    "BPCCS": [1, 2, 3, 4, 5, 6],
    "SVICS-G": [1, 2, 3]
}

SEM_LABELS = {
    1: "1 -- First Semester",
    2: "2 -- Second Semester",
    3: "3 -- Third Semester",
    4: "4 -- Fourth Semester",
    5: "5 -- Fifth Semester",
    6: "6 -- Sixth Semester",
}

# ── FEATURE ENGINEERING ────────────────────────────────────
def apply_semester_signal(base_prob, semester, sem_weight=None):
    if semester >= 4:
        return base_prob
    adaptive_weights = {1: 0.20, 2: 0.15, 3: 0.05}
    weight = adaptive_weights.get(int(semester), 0.05)
    sem_signal = SEM_HIST_RATE.get(int(semester), 0.032)
    combined = (1 - weight) * base_prob + weight * sem_signal
    return float(np.clip(combined, 0.0, 1.0))

# ── DATA & MODEL LOADING ───────────────────────────────────
@st.cache_data
def load_raw(dataset_version):
    df = PipelineStore().dataset(dataset_version)
    df["Churn Label"] = df["is_churned"].map({0: "Active", 1: "Churned"}).fillna("Pending")
    return df

@st.cache_resource
def load_model(model_version):
    if model_version:
        bundle = PipelineStore().bundle(model_version)
        return bundle["model"], bundle["scaler"], bundle["threshold"], bundle["features"], bundle["semester_rates"]
    m = joblib.load(os.path.join(FOLDER, "churn_model.pkl"))
    sc = joblib.load(os.path.join(FOLDER, "churn_scaler.pkl"))
    th = joblib.load(os.path.join(FOLDER, "churn_threshold.pkl"))
    ft = joblib.load(os.path.join(FOLDER, "churn_feature_names.pkl"))
    sw = joblib.load(os.path.join(FOLDER, "churn_sem_weight.pkl"))
    return m, sc, th, ft, sw

@st.cache_data
def get_test_set(feats_tuple, model_version, dataset_version):
    from sklearn.model_selection import train_test_split
    if model_version:
        test = PipelineStore().bundle(model_version)["test_data"]
        return rebuild_features(test)[list(feats_tuple)], test["is_churned"].astype(int), test["Current Semester"]
    test = legacy_test_data()
    return rebuild_features(test)[list(feats_tuple)], test["is_churned"].astype(int), test["Current Semester"]

try:
    pipeline_store = PipelineStore()
    dataset_version = pipeline_store.dataset_version()
    model_version = pipeline_store.active_version()
    raw = load_raw(dataset_version).dropna(subset=["is_churned"])
    model, scaler, thresh, feats, sem_weight = load_model(model_version)
    if model_version:
        SEM_HIST_RATE = sem_weight
    X_test, y_test, sem_test = get_test_set(tuple(feats), model_version, dataset_version)
    Xs_test    = scaler.transform(X_test)
    base_probs = model.predict_proba(Xs_test)[:,1]
    all_probs  = np.array([apply_semester_signal(p, s) for p, s in zip(base_probs, sem_test.values)])
    all_preds  = (all_probs >= thresh).astype(int)
    lazy_import_sklearn()
    ACTUAL_ACC = accuracy_score(y_test, all_preds)
    ACTUAL_AUC = roc_auc_score(y_test, all_probs)
    ACTUAL_F1 = f1_score(y_test, all_preds, zero_division=0)
    LOADED = True
except Exception as e:
    LOADED = False
    ERR = str(e)
    ACTUAL_ACC = ACTUAL_AUC = ACTUAL_F1 = None

# ── SIDEBAR ────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""<div style='padding:24px 18px 14px;text-align:center;
         border-bottom:2px solid rgba(200,169,110,0.4)'>
      <div style='font-size:36px;margin-bottom:6px'>&#127891;</div>
      <div style='font-size:18px;font-weight:700;color:{GOLD};letter-spacing:1px'>BCA CHURN</div>
      <div style='font-size:10px;color:#8FA3B8;margin-top:4px;letter-spacing:2.5px;
           text-transform:uppercase'>Analysis Project</div></div>""", unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    page = st.radio("Navigation",[
        "I.    Introduction","II.   Dataset Overview","III.  Data Cleaning",
        "IV.   EDA","V.    Feature Engineering","VI.   Model & Prediction",
        "VII.  Data & Training", "VIII. CSV Predictions"],
        label_visibility="collapsed")
    if LOADED:
        st.markdown("<br>",unsafe_allow_html=True)
        st.markdown(f"""<div style='font-size:11px;color:#8FA3B8;padding:0 4px;line-height:2.2'>
          <span style='color:{GOLD}'>&#9632;</span>&nbsp; {type(model).__name__}<br>
          <span style='color:{GOLD}'>&#9632;</span>&nbsp; Accuracy: {ACTUAL_ACC*100:.1f}%<br>
          <span style='color:{GOLD}'>&#9632;</span>&nbsp; AUC: {ACTUAL_AUC:.3f}<br>
          <span style='color:{GOLD}'>&#9632;</span>&nbsp; F1: {ACTUAL_F1:.3f}<br>
          <span style='color:{GOLD}'>&#9632;</span>&nbsp; Features: {len(feats)} admission<br>
          <span style='color:{GOLD}'>&#9632;</span>&nbsp; + adaptive semester signal<br>
          <span style='color:{GOLD}'>&#9632;</span>&nbsp; (S1:20%, S2:15%, S3:5%)</div>""",
            unsafe_allow_html=True)

if 'CSV Predictions' in page:
    render_batch_prediction()
    st.stop()

if "Data & Training" in page:
    render_training_page()
    st.stop()

if not LOADED:
    st.error(f"The prediction model could not be loaded: {ERR}")
    st.info("Open Data & Training in the sidebar to save historical data, train, and activate a new model.")
    st.stop()

if not model_version:
    st.warning("Legacy model: displayed scores are a reconstructed split, not verified held-out performance. "
               "Use Data & Training to train and activate a versioned model with a frozen test cohort.")
else:
    st.caption(f"Active model: {model_version} · Evaluation: frozen test cohort saved with this model. "
               "Dataset and EDA pages show labeled records only; pending records are in Data & Training.")
st.caption("Target: first-semester dropout. Class 0 includes later dropout. Narrative examples on the "
           "original analysis pages describe the 2023–24 cohort; Data & Training reports are version-specific.")

# ================================================================
#  I. INTRODUCTION
# ================================================================
if "Introduction" in page:
    st.title("Interpretable Student Churn Prediction Using Machine Learning with Adaptive Semester Weighting")
    st.markdown("<p style='font-size:15px;color:#5C6B7A;font-style:italic;margin-bottom:20px'>"
                "Predicting student dropout in BCA using machine learning -- historical and updated student outcomes.</p>",
                unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Labeled Students", len(raw)); c2.metric("No First-semester Dropout", int((raw.is_churned == 0).sum()))
    c3.metric("First-semester Dropout", int(raw.is_churned.sum())); c4.metric("Churn Rate", f"{raw.is_churned.mean():.1%}")
    st.divider()

    st.markdown(f"""<div style='background:{NAVY};border:2px solid {GOLD};padding:16px 20px;margin-bottom:20px'>
      <div style='font-size:13px;font-weight:700;color:{GOLD};text-transform:uppercase;
           letter-spacing:1px;margin-bottom:10px'>Model Highlights - {type(model).__name__}</div>
      <div style='display:grid;grid-template-columns:repeat(5,1fr);gap:10px'>
        <div style='text-align:center;background:rgba(200,169,110,0.1);padding:12px 6px'>
          <div style='font-size:24px;font-weight:700;color:{GOLD}'>{ACTUAL_ACC*100:.1f}%</div>
          <div style='font-size:9px;color:#8FA3B8;text-transform:uppercase;margin-top:3px;letter-spacing:0.5px'>Accuracy</div></div>
        <div style='text-align:center;background:rgba(200,169,110,0.1);padding:12px 6px'>
          <div style='font-size:24px;font-weight:700;color:{GOLD}'>{ACTUAL_AUC:.3f}</div>
          <div style='font-size:9px;color:#8FA3B8;text-transform:uppercase;margin-top:3px;letter-spacing:0.5px'>ROC-AUC</div></div>
        <div style='text-align:center;background:rgba(200,169,110,0.1);padding:12px 6px'>
          <div style='font-size:24px;font-weight:700;color:{GOLD}'>{ACTUAL_F1:.3f}</div>
          <div style='font-size:9px;color:#8FA3B8;text-transform:uppercase;margin-top:3px;letter-spacing:0.5px'>F1 Score</div></div>
        <div style='text-align:center;background:rgba(200,169,110,0.1);padding:12px 6px'>
          <div style='font-size:24px;font-weight:700;color:{GOLD}'>{recall_score(y_test,all_preds,zero_division=0)*100:.0f}%</div>
          <div style='font-size:9px;color:#8FA3B8;text-transform:uppercase;margin-top:3px;letter-spacing:0.5px'>Recall</div></div>
        <div style='text-align:center;background:rgba(200,169,110,0.1);padding:12px 6px'>
          <div style='font-size:24px;font-weight:700;color:{GOLD}'>32</div>
          <div style='font-size:9px;color:#8FA3B8;text-transform:uppercase;margin-top:3px;letter-spacing:0.5px'>Features</div></div>
      </div></div>""", unsafe_allow_html=True)

    col_l, col_r = st.columns([3,2])
    with col_l:
        section_header("How the Model Works")
        info_card(f"""The model combines two complementary signals:<br><br>
          <b>Admission data model (80-95%):</b> {type(model).__name__} trained on 32 features
          known at enrollment -- HSC percentage, caste, stream, district, board,
          religion, gender, and college.<br><br>
          <b>Adaptive semester signal (5-20%):</b> Applied only for Semesters 1-3:<br>
          &bull; Sem 1: 20% weight (model-version semester rate)<br>
          &bull; Sem 2: 15% weight (model-version semester rate)<br>
          &bull; Sem 3: 5% weight (model-version semester rate)<br>
          &bull; Sem 4-6: 0% weight (admission model only)<br><br>
          Combined result: <b>{ACTUAL_ACC*100:.1f}% accuracy, AUC {ACTUAL_AUC:.3f}</b>""", color=GOLD)

        section_header("College-Specific Data Structure")
        info_card(f"""<b>BPCCS (588 students):</b> Students progress through all 6 semesters.
          Churn concentrated in early semesters.<br><br>
          <b>SVICS-G (254 students):</b> Mostly in Sem 3 (221 students).
          Prediction interface shows only Sem 1-3 for SVICS-G.""", color=NAVY)

        section_header("Project Objectives")
        
        # Strategic Objectives
        st.markdown(f"""<div style='background:{NAVY};padding:12px 16px;margin-bottom:10px;border-left:4px solid {GOLD}'>
          <div style='font-size:12px;font-weight:700;color:{GOLD};text-transform:uppercase;letter-spacing:1px;margin-bottom:8px'>
            Strategic Objectives</div>
          <div style='font-size:13px;color:#EDE8DC;line-height:1.8'>
            &#9658; Help higher management in taking strategic decisions<br>
            &#9658; Support policy making and institutional design<br>
            &#9658; Provide NAAC accreditation support through data-driven insights<br>
            &#9658; Enable proactive student retention strategies
          </div>
        </div>""", unsafe_allow_html=True)
        
        # Core Objectives
        st.markdown(f"""<div style='background:#FFF;border:1px solid #D4C5A9;padding:12px 16px;margin-bottom:10px;border-left:4px solid {GREEN}'>
          <div style='font-size:12px;font-weight:700;color:{NAVY};text-transform:uppercase;letter-spacing:1px;margin-bottom:8px'>
            Core Objectives</div>
          <div style='font-size:13px;color:#2C3E50;line-height:1.8'>
            &#9658; Identify students at risk of churn early<br>
            &#9658; Predict probability of student dropout using ML models<br>
            &#9658; Identify key factors influencing churn behavior<br>
            &#9658; Develop a decision support system (DSS) for educators
          </div>
        </div>""", unsafe_allow_html=True)
        
        # Technical Objectives
        st.markdown(f"""<div style='background:#FFF;border:1px solid #D4C5A9;padding:12px 16px;margin-bottom:10px;border-left:4px solid {AMBER}'>
          <div style='font-size:12px;font-weight:700;color:{NAVY};text-transform:uppercase;letter-spacing:1px;margin-bottom:8px'>
            Technical Objectives</div>
          <div style='font-size:13px;color:#2C3E50;line-height:1.8'>
            &#9658; Perform comprehensive EDA on 842 BCA students<br>
            &#9658; Engineer 32 admission-time features with data-driven interactions<br>
            &#9658; Train an honest Random Forest: {ACTUAL_ACC*100:.1f}% accuracy, AUC {ACTUAL_AUC:.3f}<br>
            &#9658; Use adaptive semester weighting for early semesters (1-3)<br>
            &#9658; Build an interactive prediction interface for institutional use
          </div>
        </div>""", unsafe_allow_html=True)

    with col_r:
        section_header("Data Dictionary")
        st.markdown(f"""<div style='background:#FFF;border:1px solid #D4C5A9;padding:12px;max-height:450px;overflow-y:auto'>
          <table style='width:100%;font-size:10px;border-collapse:collapse'>
            <tr style='background:{NAVY};color:{GOLD};position:sticky;top:0'>
              <th style='padding:6px;text-align:left;border:1px solid #ddd'>Field Name</th>
              <th style='padding:6px;text-align:left;border:1px solid #ddd'>Type</th>
              <th style='padding:6px;text-align:left;border:1px solid #ddd'>Description</th>
            </tr>
            <tr><td style='padding:4px;border:1px solid #eee'><b>Sr No</b></td><td>Numeric</td><td>Serial number</td></tr>
            <tr style='background:#F9F9F9'><td style='padding:4px;border:1px solid #eee'><b>Roll No</b></td><td>ID</td><td>Unique student identifier</td></tr>
            <tr><td style='padding:4px;border:1px solid #eee'><b>Temp Student</b></td><td>Binary</td><td>Temporary student flag</td></tr>
            <tr style='background:#F9F9F9'><td style='padding:4px;border:1px solid #eee'><b>Student Name</b></td><td>Text</td><td>Full name of student</td></tr>
            <tr><td style='padding:4px;border:1px solid #eee'><b>First Name</b></td><td>Text</td><td>Student's first name</td></tr>
            <tr style='background:#F9F9F9'><td style='padding:4px;border:1px solid #eee'><b>Middle Name</b></td><td>Text</td><td>Student's middle name</td></tr>
            <tr><td style='padding:4px;border:1px solid #eee'><b>Last Name</b></td><td>Text</td><td>Student's last name</td></tr>
            <tr style='background:#F9F9F9'><td style='padding:4px;border:1px solid #eee'><b>Institute</b></td><td>Categorical</td><td>BPCCS or SVICS-G</td></tr>
            <tr><td style='padding:4px;border:1px solid #eee'><b>Degree</b></td><td>Categorical</td><td>BCA degree program</td></tr>
            <tr style='background:#F9F9F9'><td style='padding:4px;border:1px solid #eee'><b>Program Medium</b></td><td>Categorical</td><td>English/Gujarati medium</td></tr>
            <tr><td style='padding:4px;border:1px solid #eee'><b>Program Group</b></td><td>Categorical</td><td>Academic group classification</td></tr>
            <tr style='background:#F9F9F9'><td style='padding:4px;border:1px solid #eee'><b>Admission Year</b></td><td>Numeric</td><td>Year of admission</td></tr>
            <tr><td style='padding:4px;border:1px solid #eee'><b>Admission Semester</b></td><td>Numeric</td><td>Semester at admission (1-6)</td></tr>
            <tr style='background:#F9F9F9'><td style='padding:4px;border:1px solid #eee'><b>Admission Term</b></td><td>Categorical</td><td>Odd/Even term</td></tr>
            <tr><td style='padding:4px;border:1px solid #eee'><b>Current Semester</b></td><td>Numeric</td><td>Current semester (1-6)</td></tr>
            <tr style='background:#F9F9F9'><td style='padding:4px;border:1px solid #eee'><b>Semester</b></td><td>Numeric</td><td>Semester number</td></tr>
            <tr><td style='padding:4px;border:1px solid #eee'><b>Admission Date</b></td><td>Date</td><td>Date of admission</td></tr>
            <tr style='background:#F9F9F9'><td style='padding:4px;border:1px solid #eee'><b>Registration Date</b></td><td>Date</td><td>University registration date</td></tr>
            <tr><td style='padding:4px;border:1px solid #eee'><b>Is D2D</b></td><td>Binary</td><td>Direct to degree flag</td></tr>
            <tr style='background:#F9F9F9'><td style='padding:4px;border:1px solid #eee'><b>Gender</b></td><td>Binary</td><td>Male/Female</td></tr>
            <tr><td style='padding:4px;border:1px solid #eee'><b>Birth Date</b></td><td>Date</td><td>Date of birth</td></tr>
            <tr style='background:#F9F9F9'><td style='padding:4px;border:1px solid #eee'><b>Admission Cast</b></td><td>Categorical</td><td>Caste at admission</td></tr>
            <tr><td style='padding:4px;border:1px solid #eee'><b>Actual Caste Category</b></td><td>Categorical</td><td>OPEN, OBC, SEBC, SCST</td></tr>
            <tr style='background:#F9F9F9'><td style='padding:4px;border:1px solid #eee'><b>ACPC Rank</b></td><td>Numeric</td><td>Admission committee rank</td></tr>
            <tr><td style='padding:4px;border:1px solid #eee'><b>Gujcat Num</b></td><td>Text</td><td>Gujarat Common Admission Test number</td></tr>
            <tr style='background:#F9F9F9'><td style='padding:4px;border:1px solid #eee'><b>Mother Name</b></td><td>Text</td><td>Mother's full name</td></tr>
            <tr><td style='padding:4px;border:1px solid #eee'><b>Birth Place</b></td><td>Text</td><td>Place of birth</td></tr>
            <tr style='background:#F9F9F9'><td style='padding:4px;border:1px solid #eee'><b>Birth Place District</b></td><td>Categorical</td><td>District of birth</td></tr>
            <tr><td style='padding:4px;border:1px solid #eee'><b>Birth Place State</b></td><td>Categorical</td><td>State of birth</td></tr>
            <tr style='background:#F9F9F9'><td style='padding:4px;border:1px solid #eee'><b>Subcast</b></td><td>Categorical</td><td>Sub-caste category</td></tr>
            <tr><td style='padding:4px;border:1px solid #eee'><b>Religion</b></td><td>Categorical</td><td>Hindu, Muslim, Christian, Jain</td></tr>
            <tr style='background:#F9F9F9'><td style='padding:4px;border:1px solid #eee'><b>Mother Tongue</b></td><td>Categorical</td><td>Primary language</td></tr>
            <tr><td style='padding:4px;border:1px solid #eee'><b>Nationality</b></td><td>Categorical</td><td>Country of citizenship</td></tr>
            <tr style='background:#F9F9F9'><td style='padding:4px;border:1px solid #eee'><b>Local Address</b></td><td>Text</td><td>Current residential address</td></tr>
            <tr><td style='padding:4px;border:1px solid #eee'><b>Local City</b></td><td>Categorical</td><td>Current city</td></tr>
            <tr style='background:#F9F9F9'><td style='padding:4px;border:1px solid #eee'><b>Local Pincode</b></td><td>Numeric</td><td>Current area pincode</td></tr>
            <tr><td style='padding:4px;border:1px solid #eee'><b>Local District</b></td><td>Categorical</td><td>Current district</td></tr>
            <tr style='background:#F9F9F9'><td style='padding:4px;border:1px solid #eee'><b>Local Taluka</b></td><td>Categorical</td><td>Current taluka/tehsil</td></tr>
            <tr><td style='padding:4px;border:1px solid #eee'><b>Local State</b></td><td>Categorical</td><td>Current state</td></tr>
            <tr style='background:#F9F9F9'><td style='padding:4px;border:1px solid #eee'><b>Local Country</b></td><td>Categorical</td><td>Current country</td></tr>
            <tr><td style='padding:4px;border:1px solid #eee'><b>University Email</b></td><td>Email</td><td>Official university email</td></tr>
            <tr style='background:#F9F9F9'><td style='padding:4px;border:1px solid #eee'><b>Permanent Address</b></td><td>Text</td><td>Home address</td></tr>
            <tr><td style='padding:4px;border:1px solid #eee'><b>Permanent City</b></td><td>Categorical</td><td>Home city</td></tr>
            <tr style='background:#F9F9F9'><td style='padding:4px;border:1px solid #eee'><b>Permanent Pincode</b></td><td>Numeric</td><td>Home pincode</td></tr>
            <tr><td style='padding:4px;border:1px solid #eee'><b>Permanent District</b></td><td>Categorical</td><td>Home district (35+ districts)</td></tr>
            <tr style='background:#F9F9F9'><td style='padding:4px;border:1px solid #eee'><b>Permanent Taluka</b></td><td>Categorical</td><td>Home taluka</td></tr>
            <tr><td style='padding:4px;border:1px solid #eee'><b>Permanent State</b></td><td>Categorical</td><td>Home state</td></tr>
            <tr style='background:#F9F9F9'><td style='padding:4px;border:1px solid #eee'><b>Permanent Country</b></td><td>Categorical</td><td>Home country</td></tr>
            <tr><td style='padding:4px;border:1px solid #eee'><b>Home Phone No</b></td><td>Phone</td><td>Landline number</td></tr>
            <tr style='background:#F9F9F9'><td style='padding:4px;border:1px solid #eee'><b>Mobile No</b></td><td>Phone</td><td>Primary mobile number</td></tr>
            <tr><td style='padding:4px;border:1px solid #eee'><b>Emergency No</b></td><td>Phone</td><td>Emergency contact number</td></tr>
            <tr style='background:#F9F9F9'><td style='padding:4px;border:1px solid #eee'><b>Phone NO</b></td><td>Phone</td><td>Alternate phone number</td></tr>
            <tr><td style='padding:4px;border:1px solid #eee'><b>Email</b></td><td>Email</td><td>Personal email address</td></tr>
            <tr style='background:#F9F9F9'><td style='padding:4px;border:1px solid #eee'><b>Guardian Name</b></td><td>Text</td><td>Parent/guardian name</td></tr>
            <tr><td style='padding:4px;border:1px solid #eee'><b>Guardian Address</b></td><td>Text</td><td>Guardian's address</td></tr>
            <tr style='background:#F9F9F9'><td style='padding:4px;border:1px solid #eee'><b>Guardian City</b></td><td>Categorical</td><td>Guardian's city</td></tr>
            <tr><td style='padding:4px;border:1px solid #eee'><b>Guardian Pincode</b></td><td>Numeric</td><td>Guardian's pincode</td></tr>
            <tr style='background:#F9F9F9'><td style='padding:4px;border:1px solid #eee'><b>Guardian Mobile NO</b></td><td>Phone</td><td>Guardian's mobile</td></tr>
            <tr><td style='padding:4px;border:1px solid #eee'><b>Guardian Phone NO</b></td><td>Phone</td><td>Guardian's phone</td></tr>
            <tr style='background:#F9F9F9'><td style='padding:4px;border:1px solid #eee'><b>Guardian Email</b></td><td>Email</td><td>Guardian's email</td></tr>
            <tr><td style='padding:4px;border:1px solid #eee'><b>Bloodgroup</b></td><td>Categorical</td><td>Blood group (A+, B+, O+, etc.)</td></tr>
            <tr style='background:#F9F9F9'><td style='padding:4px;border:1px solid #eee'><b>Marital Status</b></td><td>Binary</td><td>Married/Unmarried</td></tr>
            <tr><td style='padding:4px;border:1px solid #eee'><b>Student Status</b></td><td>Categorical</td><td>Current enrollment status</td></tr>
            <tr style='background:#F9F9F9'><td style='padding:4px;border:1px solid #eee'><b>Is Handicapped</b></td><td>Binary</td><td>Disability flag</td></tr>
            <tr><td style='padding:4px;border:1px solid #eee'><b>Disability</b></td><td>Text</td><td>Type of disability if any</td></tr>
            <tr style='background:#F9F9F9'><td style='padding:4px;border:1px solid #eee'><b>Photo Upload</b></td><td>Binary</td><td>Photo uploaded flag</td></tr>
            <tr><td style='padding:4px;border:1px solid #eee'><b>Fees Pending</b></td><td>Binary</td><td>Outstanding fees flag</td></tr>
            <tr style='background:#F9F9F9'><td style='padding:4px;border:1px solid #eee'><b>Total Fees</b></td><td>Numeric</td><td>₹18,000 or ₹27,000</td></tr>
            <tr><td style='padding:4px;border:1px solid #eee'><b>Received Fees</b></td><td>Numeric</td><td>Amount paid</td></tr>
            <tr style='background:#F9F9F9'><td style='padding:4px;border:1px solid #eee'><b>Pending Fees</b></td><td>Numeric</td><td>Amount outstanding</td></tr>
            <tr><td style='padding:4px;border:1px solid #eee'><b>Use Hostel</b></td><td>Binary</td><td>Hostel accommodation flag</td></tr>
            <tr style='background:#F9F9F9'><td style='padding:4px;border:1px solid #eee'><b>Detained</b></td><td>Binary</td><td>Academic detention flag</td></tr>
            <tr><td style='padding:4px;border:1px solid #eee'><b>Last Exam</b></td><td>Categorical</td><td>HSC (833) or SSC (9)</td></tr>
            <tr style='background:#F9F9F9'><td style='padding:4px;border:1px solid #eee'><b>Seat No</b></td><td>Text</td><td>Previous exam seat number</td></tr>
            <tr><td style='padding:4px;border:1px solid #eee'><b>Last Exam Percentage</b></td><td>Continuous</td><td>Marks 0-100%</td></tr>
            <tr style='background:#F9F9F9'><td style='padding:4px;border:1px solid #eee'><b>Percentile</b></td><td>Continuous</td><td>Percentile rank</td></tr>
            <tr><td style='padding:4px;border:1px solid #eee'><b>Last Exam Pass</b></td><td>Categorical</td><td>Passing year (e.g., 2022-23)</td></tr>
            <tr style='background:#F9F9F9'><td style='padding:4px;border:1px solid #eee'><b>Last Exam Board</b></td><td>Categorical</td><td>G.H.S.E.B, CBSE, GSEB, Other</td></tr>
            <tr><td style='padding:4px;border:1px solid #eee'><b>Specialization</b></td><td>Categorical</td><td>COMMERCE, SCIENCE, ARTS</td></tr>
            <tr style='background:#F9F9F9'><td style='padding:4px;border:1px solid #eee'><b>Last Institute Name</b></td><td>Text</td><td>Previous school/college name</td></tr>
            <tr><td style='padding:4px;border:1px solid #eee'><b>Last Institute City</b></td><td>Categorical</td><td>Previous institute city</td></tr>
            <tr style='background:#F9F9F9'><td style='padding:4px;border:1px solid #eee'><b>Last Institute State</b></td><td>Categorical</td><td>Previous institute state</td></tr>
            <tr><td style='padding:4px;border:1px solid #eee'><b>Last Institute Country</b></td><td>Categorical</td><td>Previous institute country</td></tr>
            <tr style='background:#F9F9F9'><td style='padding:4px;border:1px solid #eee'><b>Adhar Number</b></td><td>ID</td><td>Aadhaar card number</td></tr>
            <tr><td style='padding:4px;border:1px solid #eee'><b>Name as per adhar</b></td><td>Text</td><td>Name on Aadhaar card</td></tr>
            <tr style='background:#F9F9F9'><td style='padding:4px;border:1px solid #eee'><b>Mobile no as per adhar</b></td><td>Phone</td><td>Aadhaar linked mobile</td></tr>
            <tr><td style='padding:4px;border:1px solid #eee'><b>B.E.D method</b></td><td>Categorical</td><td>Education method</td></tr>
            <tr style='background:#F9F9F9'><td style='padding:4px;border:1px solid #eee'><b>ABCID</b></td><td>ID</td><td>Academic Bank of Credits ID</td></tr>
            <tr><td style='padding:4px;border:1px solid #eee'><b>Admission Type</b></td><td>Categorical</td><td>Type of admission process</td></tr>
            <tr style='background:#F9F9F9'><td style='padding:4px;border:1px solid #eee'><b>Churn Student</b></td><td>Target</td><td>Dropout status (Yes/No)</td></tr>
            <tr><td style='padding:4px;border:1px solid #eee'><b>student_status_new</b></td><td>Target</td><td>active / dropout_sem1 / dropout_mid</td></tr>
          </table>
        </div>""", unsafe_allow_html=True)
        
        fig = go.Figure(go.Pie(labels=["Active (735)","Churned (107)"],values=[735,107],hole=0.6,
            marker=dict(colors=[GREEN,RED],line=dict(color="#FFF",width=3)),
            textinfo="label+percent",textfont=dict(family="Times New Roman",size=14)))
        fig.add_annotation(text="<b>842</b><br>Students",x=0.5,y=0.5,showarrow=False,
            font=dict(size=18,family="Times New Roman",color=NAVY))
        tnr(fig,320); fig.update_layout(title="Class Distribution",legend=dict(orientation="h",y=-0.06))
        st.plotly_chart(fig, use_container_width=True)

        sem_data = raw.groupby("Current Semester").agg(
            n=("is_churned","count"), c=("is_churned","sum")).reset_index()
        fig_s = go.Figure(go.Bar(
            x=[f"Sem {s}" for s in sem_data["Current Semester"]],
            y=sem_data["n"],
            text=sem_data["n"], textposition="outside",
            textfont=dict(size=14, color=NAVY), cliponaxis=False,
            marker_color=GOLD))
        tnr(fig_s,290)
        fig_s.update_layout(title="Students per semester", yaxis_title="Students", showlegend=False)
        st.plotly_chart(fig_s, use_container_width=True)

        college_sem = raw.groupby(["Institute", "Current Semester"]).size().reset_index(name="Count")
        fig_cs = px.bar(college_sem, x="Current Semester", y="Count", color="Institute",
            barmode="group", text="Count",
            color_discrete_map={"BPCCS":NAVY, "SVICS-G":GOLD},
            title="College-Semester Distribution")
        fig_cs.update_traces(textposition="outside", textfont=dict(size=14, color=NAVY), cliponaxis=False)
        tnr(fig_cs, 290)
        st.plotly_chart(fig_cs, use_container_width=True)

    st.divider()
    section_header("Project Methodology")
    steps=[("I","Data Collection","Raw records BPCCS & SVICS-G"),
           ("II","Data Cleaning","Nulls, typos, standardisation"),
           ("III","EDA","Uni & Bivariate analysis"),
           ("IV","Feature Engg.","32 admission features"),
           ("V","Model Training","RF + semester signal, 70/30"),
           ("VI","Prediction","Interactive churn UI")]
    cols = st.columns(6)
    for col,(n,t,d) in zip(cols,steps):
        col.markdown(f"""<div style='background:{NAVY};padding:14px 8px;text-align:center;
             border-top:4px solid {GOLD}'>
          <div style='font-size:18px;font-weight:700;color:{GOLD}'>{n}</div>
          <div style='font-size:10px;font-weight:700;color:#EDE8DC;margin:4px 0;text-transform:uppercase'>{t}</div>
          <div style='font-size:9.5px;color:#8FA3B8;line-height:1.5'>{d}</div></div>""",
            unsafe_allow_html=True)

# ================================================================
#  II. DATASET OVERVIEW
# ================================================================
elif "Dataset" in page:
    st.title("Dataset Overview")
    t1,t2,t3 = st.tabs(["  Column Guide  ","  Sample Data  ","  Class Balance  "])
    with t1:
        section_header("Column Reference")
        guide=[("Roll No","ID","Not used in model"),
               ("Institute","Nominal","BPCCS (588) or SVICS-G (254)"),
               ("Current Semester","Ordinal","1-6. Used as signal for semesters 1-3."),
               ("Gender","Binary","Male / Female"),
               ("Admission Cast Category","Nominal","OPEN, OBC, SEBC, SCST"),
               ("Religion","Nominal","Hindu, Muslim, Christian, Jain"),
               ("Permanent District","Nominal","Home district of student"),
               ("Total Fees","Numeric","Rs 18,000 (BPCCS) or Rs 27,000 (SVICS-G)"),
               ("Last Exam","Binary","HSC (833) or SSC (9)"),
               ("Last Exam Percentage","Continuous","0-100. Churned mean 58.7%, Active mean 61.5%"),
               ("Last Exam Passing","Ordinal","Year e.g. 2022-23"),
               ("Last Exam Board/Uni.","Nominal","G.H.S.E.B, CBSE, GSEB, Other"),
               ("Specialisation","Nominal","COMMERCE, SCIENCE, ARTS"),
               ("student_status_new","Target","active / dropout_sem1 / dropout_mid")]
        tc={"ID":"#8FA3B8","Nominal":GOLD,"Binary":GREEN,"Ordinal":AMBER,
            "Continuous":RED,"Target":NAVY,"Numeric":"#5C6B7A"}
        for c,dt,desc in guide:
            cc=tc.get(dt,"#888")
            st.markdown(f"""<div style='display:flex;align-items:center;padding:8px 12px;
                 background:#FFF;border-bottom:1px solid #EDE8DC'>
              <code style='min-width:200px;font-size:12.5px;color:{NAVY};font-weight:700'>{c}</code>
              <span style='min-width:90px;background:{cc};color:#FAFAF7;font-size:10px;
                    font-weight:700;padding:2px 7px;text-transform:uppercase'>{dt}</span>
              <span style='color:#4A5568;font-size:13px;padding-left:12px'>{desc}</span></div>""",
                unsafe_allow_html=True)
    with t2:
        c1,c2,c3=st.columns(3)
        c1.metric("Rows","842"); c2.metric("Columns","14"); c3.metric("Missing Values","0")
        st.dataframe(raw.head(20), use_container_width=True)
        st.dataframe(raw[["Current Semester","Last Exam Percentage","Total Fees","is_churned"]].describe().round(2),
                     use_container_width=True)
    with t3:
        ca,cb=st.columns(2)
        with ca:
            dd=raw["student_status_new"].value_counts().reset_index(); dd.columns=["Status","Count"]
            fig=px.bar(dd,x="Status",y="Count",color="Status",
                color_discrete_map={"active":GREEN,"dropout_sem1":RED,"dropout_mid":AMBER},
                text="Count",title="Student status breakdown")
            fig.update_traces(textposition="outside", textfont=dict(size=14, color=NAVY), cliponaxis=False)
            tnr(fig,360); fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with cb:
            info_card(f"""<b>842 total students:</b><br><br>
              <b style='color:{GREEN}'>735 Active</b> -- 87.3% retained<br>
              <b style='color:{RED}'>107 dropout_sem1</b> -- 12.7% churned (model target)<br>
              <b style='color:{AMBER}'>238 dropout_mid</b> -- also dropout but is_churned=0<br><br>
              The 87.3% vs 12.7% imbalance is handled by <code>class_weight="balanced"</code>
              in the Random Forest -- no synthetic oversampling used.""", color=GOLD)

# ================================================================
#  III. DATA CLEANING
# ================================================================
elif "Cleaning" in page:
    st.title("Data Cleaning")
    section_header("10 Cleaning Steps Applied")
    steps=[
        ("1","Null Value Check","All 14 columns had zero null values after merging source files.",GREEN),
        ("2","District Standardisation","35+ spelling variants corrected: GANDHIANAGR to GANDHINAGAR, etc.",AMBER),
        ("3","Exam Year Format Fix","Mar-23, Jul-22 converted to 2022-23, 2023-24.",AMBER),
        ("4","Board Standardisation","G.S.E.B. unified to GSEB. Rare boards merged into Other.",AMBER),
        ("5","Specialisation Typos","COMERECE, COMM, COMERCE all corrected to COMMERCE.",AMBER),
        ("6","Caste Category Merge","SC + ST merged into SCST per government reservation structure.",AMBER),
        ("7","Target Column Creation","is_churned=1 for dropout_sem1, is_churned=0 for active/dropout_mid.",NAVY),
        ("8","Redundant Column Removal","Roll No, Name, Dates removed -- no predictive value.",RED),
        ("9","Continuous Exam % Preserved","Last Exam Percentage kept as continuous float, not bucketed.",GREEN),
        ("10","Rich Feature Engineering","32 data-driven admission features including polynomial and interactions.",NAVY),
    ]
    for n,t,d,c in steps:
        st.markdown(f"""<div style='display:flex;align-items:flex-start;background:#FFF;
             border:1px solid #D4C5A9;border-left:5px solid {c};padding:13px 17px;margin-bottom:8px'>
          <div style='font-size:21px;font-weight:700;color:{c};min-width:40px;padding-top:1px'>{n}.</div>
          <div style='padding-left:12px'>
            <div style='font-weight:700;font-size:14px;color:{NAVY};margin-bottom:3px'>{t}</div>
            <div style='font-size:13px;color:#4A5568;line-height:1.6'>{d}</div>
          </div></div>""", unsafe_allow_html=True)

# ================================================================
#  IV. EDA
# ================================================================
elif "EDA" in page:
    st.title("Exploratory Data Analysis")
    st.markdown("<p style='font-size:14px;color:#5C6B7A;font-style:italic'>"
                "Charts use the current labeled dataset with semester values 1-6.</p>",
                unsafe_allow_html=True)

    section_banner("SECTION A  --  UNIVARIATE ANALYSIS")
    u1,u2,u3 = st.tabs(["  Categorical Features  ","  Numerical Features  ","  Geographic  "])

    with u1:
        section_header("Distribution of Categorical Features")
        ca,cb = st.columns(2)
        with ca:
            inst=raw["Institute"].value_counts().reset_index(); inst.columns=["Institute","Count"]
            fig=px.bar(inst,x="Institute",y="Count",text="Count",color="Institute",
                color_discrete_map={"BPCCS":NAVY,"SVICS-G":GOLD},title="Students per institute")
            fig.update_traces(textposition="outside", textfont=dict(size=14, color=NAVY), cliponaxis=False)
            tnr(fig,320); fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            chart_insight("What this chart tells us",
                "BPCCS (588, 69.8%) charges Rs 18,000. SVICS-G (254, 30.2%) charges Rs 27,000. "
                "OBC students at BPCCS churn at 19.3% vs 15.6% at SVICS-G.")
        with cb:
            gen=raw["Gender"].value_counts().reset_index(); gen.columns=["Gender","Count"]
            fig2=go.Figure(go.Pie(labels=gen["Gender"],values=gen["Count"],hole=0.5,
                marker=dict(colors=[NAVY,GOLD],line=dict(color="#FFF",width=2)),
                textinfo="label+percent",textfont=dict(family="Times New Roman",size=14)))
            tnr(fig2,320); fig2.update_layout(title="Gender distribution")
            st.plotly_chart(fig2, use_container_width=True)
            chart_insight("What this chart tells us",
                "Males make up 64.7%, females 35.3%. Female students at SVICS-G churn at 14.4% "
                "-- captured through the female_svics interaction feature.")
        ca2,cb2=st.columns(2)
        with ca2:
            cast=raw["Admission Cast Category"].value_counts().reset_index(); cast.columns=["Caste","Count"]
            fig3=px.bar(cast,x="Count",y="Caste",orientation="h",text="Count",color="Caste",
                color_discrete_sequence=[NAVY,GOLD,GREEN,RED],title="Students by caste category")
            fig3.update_traces(textposition="outside", textfont=dict(size=14, color=NAVY), cliponaxis=False)
            tnr(fig3,320); fig3.update_layout(showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)
            chart_insight("What this chart tells us",
                "OPEN caste dominates (470, 55.8%). OBC has the highest churn rate at 18.1%. "
                "SCST students show the lowest churn (8.0%), possibly due to government scholarships.")
        with cb2:
            spec=raw["Specialisation"].value_counts().reset_index(); spec.columns=["Spec","Count"]
            fig5=go.Figure(go.Pie(labels=spec["Spec"],values=spec["Count"],hole=0.5,
                marker=dict(colors=[NAVY,GOLD,GREEN],line=dict(color="#FFF",width=2)),
                textinfo="label+percent",textfont=dict(family="Times New Roman",size=14)))
            tnr(fig5,320); fig5.update_layout(title="12th stream distribution")
            st.plotly_chart(fig5, use_container_width=True)
            chart_insight("What this chart tells us",
                "Commerce dominates at 86.2%. Science stream (29, 3.4%) has the highest churn "
                "rate at 20.7% -- subject mismatch is a known dropout trigger.")
        ca3,cb3=st.columns(2)
        with ca3:
            board=raw["Last Exam Board/Uni."].value_counts().reset_index(); board.columns=["Board","Count"]
            fig6=px.bar(board,x="Count",y="Board",orientation="h",text="Count",color="Count",
                color_continuous_scale=[[0,GOLD],[1,NAVY]],title="Exam board distribution")
            fig6.update_traces(textposition="outside", textfont=dict(size=14, color=NAVY), cliponaxis=False)
            tnr(fig6,320); fig6.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig6, use_container_width=True)
            chart_insight("What this chart tells us",
                "G.H.S.E.B dominates with 812 students (96.4%). The 13 CBSE students show a "
                "23.1% churn rate -- highest of any board.")
        with cb3:
            rel=raw["Religion"].value_counts().reset_index(); rel.columns=["Religion","Count"]
            fig7=go.Figure(go.Pie(labels=rel["Religion"],values=rel["Count"],hole=0.5,
                marker=dict(colors=[NAVY,GOLD,GREEN,RED],line=dict(color="#FFF",width=2)),
                textinfo="label+percent",textfont=dict(family="Times New Roman",size=14)))
            tnr(fig7,320); fig7.update_layout(title="Religion distribution")
            st.plotly_chart(fig7, use_container_width=True)
            chart_insight("What this chart tells us",
                "Hindu students dominate at 91%. Muslim students (69, 8.2%) show a higher "
                "churn rate of 17.4% vs the 12.3% Hindu rate.")
        
        # Additional charts
        ca4,cb4=st.columns(2)
        with ca4:
            fees=raw["Total Fees"].value_counts().reset_index(); fees.columns=["Fees","Count"]
            fees["Fees"] = fees["Fees"].apply(lambda x: f"₹{x:,}")
            fig8=px.bar(fees,x="Fees",y="Count",text="Count",color="Count",
                color_continuous_scale=[[0,GOLD],[1,NAVY]],title="Total fees distribution")
            fig8.update_traces(textposition="outside", textfont=dict(size=14, color=NAVY), cliponaxis=False)
            tnr(fig8,320); fig8.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig8, use_container_width=True)
            chart_insight("What this chart tells us",
                "₹18,000 (BPCCS): 588 students (69.8%). ₹27,000 (SVICS-G): 254 students (30.2%). "
                "Higher fees don't necessarily mean lower churn - both colleges show similar ~12-13% rates.")
        with cb4:
            exam=raw["Last Exam"].value_counts().reset_index(); exam.columns=["Exam","Count"]
            fig9=px.bar(exam,x="Exam",y="Count",text="Count",color="Exam",
                color_discrete_map={"HSC":NAVY,"SSC":GOLD},title="Last exam type distribution")
            fig9.update_traces(textposition="outside", textfont=dict(size=14, color=NAVY), cliponaxis=False)
            tnr(fig9,320); fig9.update_layout(showlegend=False)
            st.plotly_chart(fig9, use_container_width=True)
            chart_insight("What this chart tells us",
                "HSC dominates with 833 students (98.9%). Only 9 students (1.1%) entered from SSC. "
                "HSC is the standard entry qualification for BCA.")
        
        ca5,cb5=st.columns(2)
        with ca5:
            passing=raw["Last Exam Passing"].value_counts().head(6).reset_index()
            passing.columns=["Year","Count"]
            fig10=px.bar(passing,x="Year",y="Count",text="Count",color="Count",
                color_continuous_scale=[[0,GOLD],[1,NAVY]],title="Last exam passing year (Top 6)")
            fig10.update_traces(textposition="outside", textfont=dict(size=14, color=NAVY), cliponaxis=False)
            tnr(fig10,320); fig10.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig10, use_container_width=True)
            chart_insight("What this chart tells us",
                "Most students passed in 2023-24 (recent graduates). Gap year students (2021-22, 2022-23) "
                "show slightly higher churn rates, captured through the year_gap feature.")
        with cb5:
            # Churn by religion
            rel_churn=raw.groupby("Religion")["is_churned"].agg(["mean","sum","count"]).reset_index()
            rel_churn.columns=["Religion","Rate","Churned","Total"]
            rel_churn["Churn %"]=(rel_churn["Rate"]*100).round(1)
            rel_churn=rel_churn.sort_values("Churn %",ascending=False)
            fig11=px.bar(rel_churn,x="Religion",y="Churn %",text="Churn %",color="Churn %",
                color_continuous_scale=[[0,GREEN],[0.5,GOLD],[1,RED]],title="Churn rate by religion")
            fig11.update_traces(texttemplate="%{text:.1f}%",textposition="outside",
                textfont=dict(size=14, color=NAVY), cliponaxis=False)
            tnr(fig11,320); fig11.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig11, use_container_width=True)
            chart_insight("What this chart tells us",
                "Muslim students show 17.4% churn (highest), Hindu 12.3%, Christian 11.1%, Jain 0%. "
                "Muslim students from non-local districts face additional challenges.")


    with u2:
        section_header("Numerical Feature Distributions")
        ca,cb=st.columns(2)
        with ca:
            fig_h=px.histogram(raw,x="Last Exam Percentage",nbins=25,
                color_discrete_sequence=[NAVY],title="Last exam % -- overall distribution")
            fig_h.add_vline(x=raw["Last Exam Percentage"].mean(),line_dash="dash",line_color=GOLD,
                annotation_text=f"Mean {raw['Last Exam Percentage'].mean():.1f}%",
                annotation_font_color=GOLD, annotation_font_size=13)
            tnr(fig_h,340); st.plotly_chart(fig_h, use_container_width=True)
            chart_insight("What this chart tells us",
                "Exam percentage is roughly bell-shaped around 61.1%. The churned student "
                "mean (58.7%) is only 2.8 points below the active student mean (61.5%).")
        with cb:
            sem_ct=raw.groupby("Current Semester").agg(
                Students=("is_churned","count"), Churned=("is_churned","sum")).reset_index()
            sem_ct["Active"]     = sem_ct["Students"] - sem_ct["Churned"]
            sem_ct["Churn Rate"] = (sem_ct["Churned"]/sem_ct["Students"]*100).round(1)
            fig_sem=go.Figure()
            fig_sem.add_trace(go.Bar(name="Active",x=[f"Sem {s}" for s in sem_ct["Current Semester"]],
                y=sem_ct["Active"],marker_color=GREEN,text=sem_ct["Active"],textposition="inside",
                textfont=dict(size=13, color="#FFF")))
            fig_sem.add_trace(go.Bar(name="Churned",x=[f"Sem {s}" for s in sem_ct["Current Semester"]],
                y=sem_ct["Churned"],marker_color=RED,text=sem_ct["Churned"],textposition="inside",
                textfont=dict(size=13, color="#FFF")))
            tnr(fig_sem,340)
            fig_sem.update_layout(barmode="stack",title="Semester distribution (real values 1-6)",
                xaxis_title="Semester", yaxis_title="Students")
            st.plotly_chart(fig_sem, use_container_width=True)

    with u3:
        section_header("Geographic Distribution")
        top_d=raw["Permanent District"].value_counts().head(15).reset_index()
        top_d.columns=["District","Count"]
        fig_geo=px.bar(top_d,x="Count",y="District",orientation="h",text="Count",color="Count",
            color_continuous_scale=[[0,GOLD],[1,NAVY]],title="Top 15 districts by student count")
        fig_geo.update_traces(textposition="outside", textfont=dict(size=14, color=NAVY), cliponaxis=False)
        tnr(fig_geo,500); fig_geo.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_geo, use_container_width=True)
        chart_insight("What this chart tells us",
            "Gandhinagar (360) and Ahmedabad (184) = 64.7% of all students. "
            "Sabarkantha (37.5%), Kutch (25%), Banaskantha (22.2%), Mehsana (20.8%), "
            "Botad (20%) all exceed the 12.7% average churn rate.")

    st.markdown("<br>", unsafe_allow_html=True)
    section_banner("SECTION B  --  BIVARIATE ANALYSIS (vs Churn)")
    b1,b2,b3 = st.tabs(["  Categorical vs Churn  ","  Numerical vs Churn  ","  Semester Deep Dive  "])

    with b1:
        section_header("Churn Rate by Category","Higher bar = more of that group dropped out")
        cat_pairs=[("Admission Cast Category","Caste vs churn rate"),
                   ("Gender","Gender vs churn rate"),
                   ("Institute","Institute vs churn rate"),
                   ("Specialisation","Stream vs churn rate"),
                   ("Last Exam Board/Uni.","Board vs churn rate"),
                   ("Last Exam","Exam type vs churn rate")]
        ca,cb=st.columns(2)
        for i,(col,title) in enumerate(cat_pairs):
            grp=raw.groupby(col)["is_churned"].agg(["mean","sum","count"]).reset_index()
            grp.columns=[col,"Rate","Churned","Total"]
            grp["Churn %"]=(grp["Rate"]*100).round(1)
            grp=grp.sort_values("Churn %",ascending=False)
            fig=px.bar(grp,x=col,y="Churn %",color="Churn %",
                color_continuous_scale=[[0,GREEN],[0.5,GOLD],[1,RED]],text="Churn %",title=title,
                hover_data={"Churned":True,"Total":True})
            fig.update_traces(texttemplate="%{text:.1f}%",textposition="outside",
                textfont=dict(size=14, color=NAVY), cliponaxis=False)
            tnr(fig,330); fig.update_layout(coloraxis_showscale=False)
            with (ca if i%2==0 else cb):
                st.plotly_chart(fig, use_container_width=True)
                if col == "Admission Cast Category":
                    chart_insight("Caste category insights",
                        "OBC: 18.1% (highest), OPEN: 12.8%, SEBC: 11.8%, SCST: 8.0% (lowest). "
                        "SCST students benefit from scholarships and support systems.")
                elif col == "Gender":
                    chart_insight("Gender insights",
                        "Male: 13.6%, Female: 11.1% overall. But female SVICS-G students show 14.4% churn.")
                elif col == "Institute":
                    chart_insight("Institute insights",
                        "BPCCS: 12.8%, SVICS-G: 12.6%. Nearly identical churn rates despite fee difference.")
                elif col == "Specialisation":
                    chart_insight("Stream insights",
                        "Science: 20.7% (highest - subject mismatch), Arts: 14.9%, Commerce: 12.1% (best fit).")
                elif col == "Last Exam Board/Uni.":
                    chart_insight("Board insights",
                        "CBSE: 23.1% (highest - adjustment issues), GSEB: 14.3%, G.H.S.E.B: 12.3%.")
                elif col == "Last Exam":
                    chart_insight("Exam type insights",
                        "HSC: 12.7%, SSC: 11.1%. No significant difference, but sample size for SSC is very small (9 students).")

        section_header("Institute x Caste x Churn -- Sunburst")
        grp_sun=raw.groupby(["Institute","Admission Cast Category","Churn Label"]).size().reset_index(name="Count")
        fig_sun=px.sunburst(grp_sun,path=["Institute","Admission Cast Category","Churn Label"],
            values="Count",color="Churn Label",
            color_discrete_map={"Active":GREEN,"Churned":RED},title="Institute > Caste > Churn")
        tnr(fig_sun,480); st.plotly_chart(fig_sun, use_container_width=True)
        chart_insight("How to read the sunburst",
            "Inner ring = Institute (BPCCS/SVICS-G). Middle ring = Caste category. "
            "Outer ring = Active (green) vs Churned (red). The proportion of red shows churn rate. "
            "OBC at BPCCS shows the largest red slice (19.3% churn).")

    with b2:
        section_header("Exam Percentage vs Churn")
        ca,cb=st.columns(2)
        with ca:
            fig_bx=px.box(raw,x="Churn Label",y="Last Exam Percentage",color="Churn Label",
                color_discrete_map={"Active":GREEN,"Churned":RED},
                title="Exam % by churn status",points="outliers")
            fig_bx.update_layout(showlegend=False)
            tnr(fig_bx,350)
            st.plotly_chart(fig_bx, use_container_width=True)
            chart_insight("What this box plot tells us",
                "Churned students have lower median (~58%) vs active (~61%), but large overlap. "
                "The model uses polynomial features (pct_dev_sq) to capture the non-linear relationship.")
        with cb:
            brackets=["0-45%","45-50%","50-55%","55-60%","60-65%","65-70%","70-75%","75%+"]
            rates=[17.6,9.3,14.2,17.2,15.5,8.6,8.0,10.6]
            fig_br=go.Figure(go.Bar(x=brackets,y=rates,
                marker_color=[RED if r>15 else AMBER if r>12 else GREEN for r in rates],
                text=[f"{r}%" for r in rates],textposition="outside",
                textfont=dict(size=14, color=NAVY), cliponaxis=False))
            fig_br.add_hline(y=12.7,line_dash="dash",line_color=GOLD,
                annotation_text="Overall 12.7%",annotation_font_color=GOLD, annotation_font_size=13)
            tnr(fig_br,350)
            fig_br.update_layout(title="Churn rate by exam % bracket -- non-linear!",
                yaxis_title="Churn Rate %",xaxis_title="HSC % Range")
            st.plotly_chart(fig_br, use_container_width=True)
            chart_insight("The non-linear exam percentage finding",
                "Counter-intuitive: 55-60% bracket has 17.2% churn -- higher than below-45% (17.6%)! "
                "The 50-65% 'middle zone' churns at 15-17%. These students have uncertain academic confidence. "
                "Very high scorers (75%+) still churn at 10.6% -- possibly overconfidence or wrong course choice.")
        
        # Additional numerical analysis
        ca2,cb2=st.columns(2)
        with ca2:
            # Fees vs Churn
            fees_churn=raw.groupby("Total Fees")["is_churned"].agg(["mean","sum","count"]).reset_index()
            fees_churn.columns=["Fees","Rate","Churned","Total"]
            fees_churn["Churn %"]=(fees_churn["Rate"]*100).round(1)
            fees_churn["Fees Label"] = fees_churn["Fees"].apply(lambda x: f"₹{x:,}")
            fig_fc=px.bar(fees_churn,x="Fees Label",y="Churn %",text="Churn %",color="Churn %",
                color_continuous_scale=[[0,GREEN],[0.5,GOLD],[1,RED]],title="Churn rate by fees")
            fig_fc.update_traces(texttemplate="%{text:.1f}%",textposition="outside",
                textfont=dict(size=14, color=NAVY), cliponaxis=False)
            tnr(fig_fc,350); fig_fc.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_fc, use_container_width=True)
            chart_insight("Fees vs churn insight",
                "₹18,000 (BPCCS): 12.8% churn. ₹27,000 (SVICS-G): 12.6% churn. "
                "Nearly identical rates suggest fees alone don't determine dropout. "
                "Other factors (caste, stream, district) matter more.")
        with cb2:
            # District churn rates (top 10)
            dist_churn=raw.groupby("Permanent District")["is_churned"].agg(["mean","sum","count"]).reset_index()
            dist_churn.columns=["District","Rate","Churned","Total"]
            dist_churn["Churn %"]=(dist_churn["Rate"]*100).round(1)
            dist_churn=dist_churn[dist_churn["Total"]>=10].sort_values("Churn %",ascending=False).head(10)
            fig_dc=px.bar(dist_churn,x="Churn %",y="District",orientation="h",text="Churn %",color="Churn %",
                color_continuous_scale=[[0,GREEN],[0.5,GOLD],[1,RED]],title="Top 10 high-risk districts")
            fig_dc.update_traces(texttemplate="%{text:.1f}%",textposition="outside",
                textfont=dict(size=13, color=NAVY), cliponaxis=False)
            tnr(fig_dc,350); fig_dc.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_dc, use_container_width=True)
            chart_insight("Geographic risk zones",
                "Sabarkantha: 37.5% (highest), Kutch: 25%, Banaskantha: 22.2%, Mehsana: 20.8%. "
                "All far from college locations. Distance matters -- local students (Gandhinagar, Ahmedabad) "
                "churn at only 11.4%.")

    with b3:
        section_header("Semester-Wise Analysis (Real Data)")
        info_card(f"""<b style='color:{NAVY}'>Note:</b> These charts use actual semester values 1-6 from the current labeled dataset,
                  showing the true distribution of students and churn across semesters.""", color=GOLD)
        sem_d=raw.groupby("Current Semester").agg(
            Students=("is_churned","count"),Churned=("is_churned","sum")).reset_index()
        sem_d["Active"]     = sem_d["Students"] - sem_d["Churned"]
        sem_d["Churn Rate"] = (sem_d["Churned"]/sem_d["Students"]*100).round(1)
        sem_d["Label"]      = ["Sem "+str(s) for s in sem_d["Current Semester"]]
        ca,cb=st.columns(2)
        with ca:
            fig_s1=go.Figure()
            fig_s1.add_trace(go.Bar(name="Active",x=sem_d["Label"],y=sem_d["Active"],
                marker_color=GREEN,text=sem_d["Active"],textposition="inside",
                textfont=dict(size=13, color="#FFF")))
            fig_s1.add_trace(go.Bar(name="Churned",x=sem_d["Label"],y=sem_d["Churned"],
                marker_color=RED,text=sem_d["Churned"],textposition="inside",
                textfont=dict(size=13, color="#FFF")))
            tnr(fig_s1,360); fig_s1.update_layout(barmode="stack",title="Students per semester")
            st.plotly_chart(fig_s1, use_container_width=True)
            chart_insight("Student distribution by semester",
                "Sem 1: 61 students (100% churned). Sem 2: 47 students (80.9% churned). "
                "Sem 3: 249 students (3.2% churned). Sem 4-6: 485 students (0% churned). "
                "Clear pattern: churn is almost entirely a Semester 1-2 phenomenon.")
        with cb:
            fig_s2=go.Figure(go.Bar(x=sem_d["Label"],y=sem_d["Churn Rate"],
                marker_color=[RED if r>50 else AMBER if r>12 else GREEN for r in sem_d["Churn Rate"]],
                text=[f"{r}%" for r in sem_d["Churn Rate"]],textposition="outside",
                textfont=dict(size=14, color=NAVY), cliponaxis=False))
            fig_s2.add_hline(y=12.7,line_dash="dash",line_color=GOLD,
                annotation_text="Overall avg 12.7%",annotation_font_color=GOLD, annotation_font_size=13)
            tnr(fig_s2,360); fig_s2.update_layout(title="Churn rate per semester")
            st.plotly_chart(fig_s2, use_container_width=True)
            chart_insight("Churn rate pattern",
                "Sem 1: 100% (all 61 students churned). Sem 2: 80.9% (38 of 47 churned). "
                "Sem 3: 3.2% (8 of 249 churned). Sem 4-6: 0% (0 of 485 churned). "
                "Once students pass Sem 2, dropout probability drops to near zero.")
        
        st.dataframe(sem_d[["Label","Students","Churned","Active","Churn Rate"]],
                     use_container_width=True, hide_index=True)
        
        chart_insight("Why adaptive semester weighting works",
            "The model uses these historical rates as signals:<br>"
            "• Sem 1: 20% weight (critical risk period)<br>"
            "• Sem 2: 15% weight (high risk period)<br>"
            "• Sem 3: 5% weight (low risk period)<br>"
            "• Sem 4-6: 0% weight (stable period - admission model only)<br><br>"
            "This adaptive approach is honest -- it doesn't over-rely on semester as a 'cheat code' "
            "but uses it as a validated contextual signal for early semesters only.")

# ================================================================
#  V. FEATURE ENGINEERING
# ================================================================
elif "Feature Engineering" in page:
    st.title("Feature Engineering")
    t1,t2,t3 = st.tabs(["  All 32 Features  ","  Feature Importance  ","  Correlation  "])
    with t1:
        section_header("32 Admission-Time Features","Known at enrollment + semester as signal for S1-3")
        feat_list=[
            ("exam_pct","Continuous","Raw HSC %","Strongest single continuous predictor."),
            ("pct_sq","Polynomial","(exam_pct/100)^2","Captures non-linear U-shape."),
            ("pct_dev","Deviation","exam_pct - 61.5","Signed deviation from active student mean."),
            ("pct_dev_sq","Polynomial","(exam_pct-61.5)^2","Symmetric risk both directions."),
            ("pct_danger","Binary","1 if 50<=pct<65","Counter-intuitive high-churn zone (15-17%)."),
            ("pct_very_low","Binary","1 if pct<45","Very low scorers: 17.6% churn rate."),
            ("gender","Binary","0=Male, 1=Female","Female generally lower risk except at SVICS-G."),
            ("college","Binary","0=BPCCS, 1=SVICS-G","Institute identifier."),
            ("fees","Continuous","Rs 18,000 or Rs 27,000","Proxy for institute type."),
            ("year_gap","Ordinal","Years since 12th (1-5)","Gap year students show slightly higher churn."),
            ("cast_obc/scst/sebc/open","One-Hot","One flag per caste","OBC=18.1%, SCST=8.0%."),
            ("rel_muslim/hindu","One-Hot","Religion flags","Muslim=17.4% churn vs Hindu=12.3%."),
            ("spec_science/arts/commerce","One-Hot","Stream flags","Science=20.7%, Arts=14.9%."),
            ("board_cbse/gseb","One-Hot","Board flags","CBSE=23.1% churn."),
            ("dist_high","Binary","1 if high-churn district","5 districts with 20-37% churn."),
            ("dist_local","Binary","1 if Gandhinagar/Ahmedabad","Local students: 11.4% churn."),
            ("pct_x_obc","Interaction","exam_pct x cast_obc","Performance x OBC flag."),
            ("pct_x_science","Interaction","exam_pct x spec_science","Science + low marks = compounded risk."),
            ("pct_x_dist","Interaction","exam_pct x dist_high","High-risk district + weak marks."),
            ("pct_x_college","Interaction","exam_pct x college","Performance x institute."),
            ("bpccs_obc","Interaction","(1-college) x cast_obc","BPCCS OBC: 19.3% churn."),
            ("female_svics","Interaction","gender x college","Female x SVICS-G: 14.4%."),
            ("obc_science","Interaction","cast_obc x spec_science","OBC Science: double vulnerability."),
            ("risk_score","Composite","Sum of 7 risk flags","Aggregate risk index (0-5)."),
            ("risk_x_pct","Interaction","risk_score x exam_pct","Top feature: combined risk x performance."),
        ]
        for feat,ftype,formula,reason in feat_list:
            color=GOLD if "Interaction" in ftype or "Composite" in ftype or "Polynomial" in ftype else NAVY
            st.markdown(f"""<div style='background:#FFF;border:1px solid #D4C5A9;border-left:5px solid {color};
                 padding:10px 15px;margin-bottom:6px'>
              <div style='display:flex;align-items:center;justify-content:space-between;margin-bottom:3px'>
                <code style='font-size:13px;font-weight:700;color:{NAVY}'>{feat}</code>
                <span style='background:{color};color:#FAFAF7;font-size:10px;font-weight:700;
                      padding:2px 8px;text-transform:uppercase'>{ftype}</span></div>
              <div style='background:#F5F2EB;padding:3px 8px;font-size:12px;color:{NAVY};margin-bottom:3px'>{formula}</div>
              <div style='font-size:12px;color:#5C6B7A'>{reason}</div></div>""",
                unsafe_allow_html=True)

    with t2:
        section_header(f"Feature Importance -- {type(model).__name__}")
        if hasattr(model,'feature_importances_'):
            fi=pd.DataFrame({"Feature":list(feats),"Importance":model.feature_importances_}
                           ).sort_values("Importance",ascending=True)
            fig_fi=go.Figure(go.Bar(x=fi["Importance"],y=fi["Feature"],orientation="h",
                marker_color=[GOLD if v>0.05 else NAVY for v in fi["Importance"]],
                text=[f"{v:.3f}" for v in fi["Importance"]],textposition="outside",
                textfont=dict(size=13, color=NAVY), cliponaxis=False))
            tnr(fig_fi,620)
            fig_fi.update_layout(title="Feature importances (base admission model)",
                xaxis_title="Importance",yaxis=dict(tickfont=dict(size=10)))
            st.plotly_chart(fig_fi, use_container_width=True)

    with t3:
        section_header("Correlation Heatmap")
        raw2=raw.copy()
        raw2['exam_pct'] =raw2['Last Exam Percentage'].fillna(raw2['Last Exam Percentage'].median())
        raw2['cast_obc'] =(raw2['Admission Cast Category']=='OBC').astype(int)
        raw2['spec_sci'] =(raw2['Specialisation']=='SCIENCE').astype(int)
        raw2['dist_hc']  =raw2['Permanent District'].isin(HIGH_CHURN_DISTS).astype(int)
        raw2['rel_mus']  =(raw2['Religion']=='Muslim').astype(int)
        raw2['college']  =(raw2['Institute']=='SVICS-G').astype(int)
        raw2['gender_n'] =(raw2['Gender']=='Female').astype(int)
        corr_df=raw2[["is_churned","exam_pct","cast_obc","spec_sci","dist_hc","rel_mus","college","gender_n"]].corr().round(2)
        corr_df.index=["Churned","Exam %","OBC","Science","High-churn dist","Muslim","SVICS-G","Female"]
        corr_df.columns=corr_df.index
        fig_c=px.imshow(corr_df,text_auto=True,color_continuous_scale="RdBu_r",
            zmin=-1,zmax=1,aspect="auto",title="Pearson correlation -- key admission features")
        fig_c.update_traces(textfont_size=11); tnr(fig_c,500)
        st.plotly_chart(fig_c, use_container_width=True)

# ================================================================
#  VI. MODEL & PREDICTION
# ================================================================
elif "Model" in page:
    st.title("Model & Prediction")
    st.markdown(f"""<p style='font-size:14px;color:#5C6B7A;font-style:italic'>
      {type(model).__name__} | 32 features + adaptive semester signal |
      Accuracy {ACTUAL_ACC*100:.1f}% | AUC {ACTUAL_AUC:.3f} | F1 {ACTUAL_F1:.3f} | saved evaluation cohort</p>""",
        unsafe_allow_html=True)
    t1,t2,t3=st.tabs(["  Model Performance  ","  Predict a Student  ","  All Test Predictions  "])

    with t1:
        acc =accuracy_score(y_test,all_preds)
        f1  =f1_score(y_test,all_preds,zero_division=0)
        auc =roc_auc_score(y_test,all_probs)
        cm  =confusion_matrix(y_test,all_preds)
        tn,fp,fn,tp_v=cm.ravel()
        prec=precision_score(y_test,all_preds,zero_division=0)
        rec =recall_score(y_test,all_preds,zero_division=0)

        section_header("Evaluation Metrics -- Saved Test Cohort")
        m1,m2,m3,m4,m5=st.columns(5)
        m1.metric("Accuracy",f"{acc*100:.1f}%"); m2.metric("ROC-AUC",f"{auc:.3f}")
        m3.metric("F1 Score",f"{f1:.3f}");       m4.metric("Recall",f"{rec:.1%}")
        m5.metric("Threshold",f"{thresh:.3f}")

        ca,cb=st.columns(2)
        with ca:
            section_header("Confusion Matrix")
            fig_cm=go.Figure(go.Heatmap(z=cm,x=["Pred: Active","Pred: Churned"],
                y=["Actual: Active","Actual: Churned"],
                text=cm,texttemplate="<b>%{text}</b>",textfont={"size":28},
                colorscale=[[0,"#F0ECE4"],[1,NAVY]],
                hovertemplate="Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>"))
            fig_cm.update_layout(
                title="",
                xaxis_title="Predicted Class",
                yaxis_title="Actual Class"
            )
            tnr(fig_cm,320); st.plotly_chart(fig_cm, use_container_width=True)
            st.markdown(f"""<div style='display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-top:7px'>
              <div style='background:rgba(45,106,79,0.11);border-left:4px solid {GREEN};padding:10px;text-align:center'>
                <div style='font-size:24px;font-weight:700;color:{GREEN}'>{tp_v}</div>
                <div style='font-size:10px;color:{GREEN};text-transform:uppercase'>TP -- Caught</div></div>
              <div style='background:rgba(139,37,37,0.1);border-left:4px solid {RED};padding:10px;text-align:center'>
                <div style='font-size:24px;font-weight:700;color:{RED}'>{fn}</div>
                <div style='font-size:10px;color:{RED};text-transform:uppercase'>FN -- Missed</div></div>
              <div style='background:rgba(155,106,26,0.1);border-left:4px solid {AMBER};padding:10px;text-align:center'>
                <div style='font-size:24px;font-weight:700;color:{AMBER}'>{fp}</div>
                <div style='font-size:10px;color:{AMBER};text-transform:uppercase'>FP -- False Alarm</div></div>
              <div style='background:rgba(28,43,58,0.09);border-left:4px solid {NAVY};padding:10px;text-align:center'>
                <div style='font-size:24px;font-weight:700;color:{NAVY}'>{tn}</div>
                <div style='font-size:10px;color:{NAVY};text-transform:uppercase'>TN -- Correct</div></div></div>""",
                unsafe_allow_html=True)
        with cb:
            section_header("Score Distribution")
            df_p=pd.DataFrame({"Score":all_probs,
                "Actual":["Churned" if v==1 else "Active" for v in y_test.values]})
            fig_pd=px.histogram(df_p,x="Score",color="Actual",nbins=30,
                barmode="overlay",opacity=0.75,
                color_discrete_map={"Active":GREEN,"Churned":RED},
                labels={"Score":"Churn Probability Score", "count":"Count"})
            fig_pd.add_vline(x=thresh,line_dash="dash",line_color=GOLD,
                annotation_text=f"Threshold {thresh:.2f}",annotation_font_color=GOLD,
                annotation_font_size=13)
            fig_pd.update_layout(
                title="",
                xaxis_title="Churn Probability Score",
                yaxis_title="Count",
                showlegend=True
            )
            tnr(fig_pd,300); st.plotly_chart(fig_pd, use_container_width=True)
            section_header("Classification Report")
            rep=classification_report(y_test,all_preds,output_dict=True,zero_division=0)
            rep_df=pd.DataFrame(rep).T.reset_index(); rep_df.columns=["Class","Precision","Recall","F1","Support"]
            rep_df=rep_df[rep_df["Class"].isin(["0","1","macro avg","weighted avg"])]
            rep_df["Class"]=rep_df["Class"].map(
                {"0":"Active","1":"Churned","macro avg":"Macro Avg","weighted avg":"Weighted Avg"})
            st.dataframe(rep_df.round(3), use_container_width=True, hide_index=True)

        section_header("ROC Curve")
        fpr_r,tpr_r,_=roc_curve(y_test,all_probs)
        fig_roc=go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr_r,y=tpr_r,
            name=f"Combined model (AUC={auc:.3f})",
            line=dict(color=NAVY,width=2.5),fill="tozeroy",fillcolor=rgba(NAVY,0.09)))
        fig_roc.add_trace(go.Scatter(x=[0,1],y=[0,1],name="Random baseline (AUC=0.500)",
            line=dict(color=GOLD,dash="dash",width=1.5)))
        fig_roc.update_layout(title=f"ROC Curve -- AUC {auc:.3f}",
            xaxis_title="False Positive Rate",yaxis_title="True Positive Rate")
        tnr(fig_roc,400); st.plotly_chart(fig_roc, use_container_width=True)

        st.divider()
        section_header("SHAP Analysis - Model Interpretability",
                      "Understanding how features impact predictions")

        with st.expander("What is SHAP?", expanded=False):
            st.markdown(f"""<div style='font-size:13px;line-height:1.7;color:{NAVY}'>
                <b>SHAP (SHapley Additive exPlanations)</b> helps explain how individual features influence the model's prediction.<br><br>
                <b>Beeswarm Plot:</b> Each dot represents one student. The horizontal position shows whether a feature pushes the prediction more toward churn or toward active.<br><br>
                <b>Positive SHAP value:</b> pushes prediction toward churn.<br>
                <b>Negative SHAP value:</b> pushes prediction toward active.<br><br>
                This makes the Random Forest model easier to understand instead of treating it only as a black-box model.
            </div>""", unsafe_allow_html=True)

        if st.checkbox("Show SHAP explanations", key="show_shap"):
            try:
                with st.spinner("Computing SHAP values..."):
                    import shap
                    import matplotlib.pyplot as plt

                    # ---------------------------------------------------------
                    # 1. Take a sample from test data
                    # ---------------------------------------------------------
                    sample_size = min(150, len(X_test))

                    sample_indices = np.random.RandomState(42).choice(
                        len(X_test),
                        sample_size,
                        replace=False
                    )

                    X_sample = X_test.iloc[sample_indices].copy()


                    # ---------------------------------------------------------
                    # 2. Scale data exactly as model expects
                    # ---------------------------------------------------------
                    X_sample_scaled = scaler.transform(X_sample)

                    # Convert back to DataFrame so feature names are preserved
                    X_sample_scaled_df = pd.DataFrame(
                        X_sample_scaled,
                        columns=X_sample.columns,
                        index=X_sample.index
                    )


                    # ---------------------------------------------------------
                    # 3. Create SHAP TreeExplainer
                    # ---------------------------------------------------------
                    if hasattr(model, "coef_"):
                        explainer = shap.LinearExplainer(model, X_sample_scaled_df)
                    else:
                        explainer = shap.TreeExplainer(model)

                    shap_values = explainer.shap_values(X_sample_scaled_df)


                    # ---------------------------------------------------------
                    # 4. FIX SHAP OUTPUT SHAPE
                    # ---------------------------------------------------------

                    # Older SHAP:
                    # [class_0_array, class_1_array]
                    if isinstance(shap_values, list):

                        if len(shap_values) == 2:
                            shap_values_churn = shap_values[1]
                        else:
                            shap_values_churn = shap_values[0]

                    else:
                        shap_values = np.asarray(shap_values)

                        # Newer SHAP RandomForest output:
                        # (samples, features, classes)
                        if shap_values.ndim == 3:

                            # Select class 1 = Churned
                            if shap_values.shape[2] >= 2:
                                shap_values_churn = shap_values[:, :, 1]
                            else:
                                shap_values_churn = shap_values[:, :, 0]

                        elif shap_values.ndim == 2:

                            shap_values_churn = shap_values

                        else:
                            raise ValueError(
                                f"Unexpected SHAP output shape: {shap_values.shape}"
                            )


                    # Force final SHAP matrix to proper 2D shape
                    shap_values_churn = np.asarray(shap_values_churn)

                    if shap_values_churn.ndim != 2:
                        shap_values_churn = np.squeeze(shap_values_churn)

                    if shap_values_churn.ndim != 2:
                        raise ValueError(
                            f"SHAP values could not be converted to 2D. "
                            f"Current shape: {shap_values_churn.shape}"
                        )


                    # ---------------------------------------------------------
                    # 5. SHAP BEESWARM / SUMMARY PLOT
                    # ---------------------------------------------------------
                    st.markdown(
                        f"""
                        <div style='font-size:14px;font-weight:600;
                        color:{NAVY};margin:15px 0 10px 0'>
                        SHAP Beeswarm Plot - Top 15 Features
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown("""
                    <div style='font-size:12px;color:#666;margin-bottom:10px'>
                        Each dot = one student |
                        Left/Right position = impact on prediction |
                        Color = feature value
                    </div>
                    """, unsafe_allow_html=True)


                    fig_shap = plt.figure(figsize=(10, 8))

                    shap.summary_plot(
                        shap_values_churn,
                        X_sample_scaled_df,
                        feature_names=list(X_sample.columns),
                        plot_type="dot",
                        max_display=15,
                        show=False
                    )

                    plt.xlabel(
                        "SHAP Value (Impact on Churn Prediction)",
                        fontsize=11
                    )

                    plt.tight_layout()

                    st.pyplot(plt.gcf(), use_container_width=True)

                    plt.close("all")


                    # ---------------------------------------------------------
                    # 6. CALCULATE MEAN ABSOLUTE SHAP IMPORTANCE
                    # ---------------------------------------------------------
                    mean_shap = np.mean(
                        np.abs(shap_values_churn),
                        axis=0
                    )

                    # IMPORTANT:
                    # Convert to a flat 1D array
                    mean_shap = np.asarray(mean_shap).flatten()

                    feature_names = list(X_sample.columns)


                    # Check dimensions before making DataFrame
                    if len(mean_shap) != len(feature_names):
                        raise ValueError(
                            f"Feature count mismatch: "
                            f"{len(feature_names)} feature names but "
                            f"{len(mean_shap)} SHAP importance values."
                        )


                    shap_importance = pd.DataFrame({
                        "Feature": feature_names,
                        "Importance": mean_shap
                    })

                    shap_importance = (
                        shap_importance
                        .sort_values("Importance", ascending=False)
                        .head(10)
                    )


                    # ---------------------------------------------------------
                    # 7. TOP 10 SHAP FEATURE IMPORTANCE
                    # ---------------------------------------------------------
                    st.markdown(
                        f"""
                        <div style='font-size:14px;font-weight:600;
                        color:{NAVY};margin:20px 0 10px 0'>
                        Top 10 Most Important Features by Mean |SHAP|
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


                    fig_importance = go.Figure(
                        go.Bar(
                            x=shap_importance["Importance"],
                            y=shap_importance["Feature"],
                            orientation="h",
                            marker=dict(
                                color=shap_importance["Importance"],
                                colorscale=[
                                    [0, GREEN],
                                    [0.5, AMBER],
                                    [1, RED]
                                ],
                                showscale=False
                            ),
                            text=shap_importance["Importance"].round(4),
                            textposition="outside",
                            textfont=dict(size=11)
                        )
                    )

                    fig_importance.update_layout(
                        title="",
                        xaxis_title="Mean |SHAP Value| (Average Impact on Prediction)",
                        yaxis_title="",
                        yaxis=dict(
                            categoryorder="total ascending",
                            title=""
                        ),
                        height=430,
                        showlegend=False,
                        margin=dict(
                            l=180,
                            r=70,
                            t=30,
                            b=50
                        )
                    )

                    tnr(fig_importance, 430)

                    st.plotly_chart(
                        fig_importance,
                        use_container_width=True
                    )


                    # ---------------------------------------------------------
                    # 8. SIMPLE EXPLANATION
                    # ---------------------------------------------------------
                    chart_insight(
                        "How to Read SHAP Analysis",
                        "Features with larger Mean |SHAP| values have a greater "
                        "overall influence on the Random Forest predictions. "
                        "In the beeswarm plot, values on the positive side push "
                        "the model more toward churn, while values on the negative "
                        "side push the prediction more toward active status."
                    )


            except Exception as e:

                st.warning(f"SHAP explanations are unavailable in this environment: {e}. Predictions remain available.")

                import traceback

                with st.expander("Explanation dependency details"):
                    st.code(traceback.format_exc())

                st.info(
                    "Install compatible SHAP dependencies to view explanations for the selected model."
                )
    with t2:
        section_header("Predict Churn Risk for a Student",
                        "32 admission features + adaptive semester signal")

        # College selection outside form for dynamic semester update
        st.markdown(f"""<div style='background:{NAVY};color:{GOLD};font-size:12px;font-weight:700;
             letter-spacing:1.2px;padding:9px 14px;text-transform:uppercase;margin-bottom:8px'>
          SELECT COLLEGE FIRST</div>""", unsafe_allow_html=True)
        
        college = st.selectbox("College",["BPCCS  (Rs.18,000)","SVICS-G  (Rs.27,000)"],key="college_select_main")
        college_key = "BPCCS" if "BPCCS" in college else "SVICS-G"
        available_sems = COLLEGE_SEMESTERS[college_key]
        
        st.markdown("<br>", unsafe_allow_html=True)

        with st.form("predict_form"):
            st.markdown(f"""<div style='background:{NAVY};color:{GOLD};font-size:12px;font-weight:700;
                 letter-spacing:1.2px;padding:9px 14px;text-transform:uppercase;margin-bottom:4px'>
              STUDENT PROFILE -- FILL ALL FIELDS</div>""", unsafe_allow_html=True)

            form_label("Institute & Academic Background")
            g1a,g1b,g1c=st.columns(3)
            # Display selected college (read-only)
            g1a.markdown(f"""<div style='font-size:13px;color:#666;margin-bottom:4px'>Selected College</div>
                <div style='background:#F0F0F0;padding:8px;border-radius:4px;font-weight:600;color:{NAVY}'>{college}</div>""", unsafe_allow_html=True)
            year_gap  =g1b.selectbox("Years Since 12th",[1,2,3,4,5])
            exam_pct  =g1c.number_input("HSC Percentage (%)",min_value=0.0,max_value=100.0,
                value=61.5,step=0.5)
            actual_fees = st.number_input("Total Fees", min_value=0.0, max_value=10000000.0,
                value=18000.0 if college_key == "BPCCS" else 27000.0, step=500.0,
                key=f"prediction_fees_{college_key}")

            sc1,sc2=st.columns([1,2])
            semester=sc1.selectbox("Current Semester",options=available_sems,
                format_func=lambda x: SEM_LABELS[x])
            sem_clr={1:RED,2:AMBER,3:GREEN,4:GREEN,5:GREEN,6:GREEN}[semester]
            sem_msg={
                1:"Semester 1 -- 20% semester signal; rate stored with the active model.",
                2:"Semester 2 -- 15% semester signal; rate stored with the active model.",
                3:"Semester 3 -- 5% semester signal; rate stored with the active model.",
                4:"Semester 4 -- Stable. Only admission model used.",
                5:"Semester 5 -- Stable. Only admission model used.",
                6:"Semester 6 -- Stable. Only admission model used."
            }[semester]
            sc2.markdown(f"""<div style='background:#FFF;border-left:4px solid {sem_clr};
                 padding:10px 14px;margin-top:4px;font-size:13px;color:{sem_clr};font-weight:600'>
              {sem_msg}</div>""", unsafe_allow_html=True)

            form_label("Demographics")
            g2a,g2b,g2c,g2d=st.columns(4)
            gender   =g2a.selectbox("Gender",["Male","Female"])
            caste    =g2b.selectbox("Caste",["OPEN","OBC","SEBC","SCST"])
            religion =g2c.selectbox("Religion",["Hindu","Muslim","Jain","Christian"])
            district =g2d.selectbox("District",ALL_DISTRICTS)

            form_label("12th Exam Details")
            g3a,g3b=st.columns(2)
            spec =g3a.selectbox("12th Stream",["COMMERCE","SCIENCE","ARTS"])
            board=g3b.selectbox("Board",["G.H.S.E.B","GSEB","CBSE","Other"])

            st.markdown("<br>",unsafe_allow_html=True)
            submitted=st.form_submit_button("PREDICT CHURN RISK")

        if submitted:
            adm_row  = build_admission_features(exam_pct,gender,college,year_gap,
                                                spec,board,caste,religion,district)
            adm_row["fees"] = actual_fees
            inp      = pd.DataFrame([adm_row])[list(feats)]
            inps     = scaler.transform(inp)
            base_p   = float(model.predict_proba(inps)[0][1])
            adaptive_weights = {1: 0.20, 2: 0.15, 3: 0.05}
            current_weight = adaptive_weights.get(semester, 0.0) if semester <= 3 else 0.0
            if semester <= 3:
                sem_signal = SEM_HIST_RATE.get(semester, 0.032)
                final_p = (1 - current_weight) * base_p + current_weight * sem_signal
                final_p = float(np.clip(final_p, 0.0, 1.0))
            else:
                final_p = base_p
            pred     = int(final_p >= thresh)
            rlbl,rcol= get_risk(final_p)
            badge    = "FLAGGED AS CHURN RISK" if pred else "PREDICTED AS ACTIVE"
            bcol     = RED if pred else GREEN

            st.markdown(f"""<div style='background:#FFF;border:3px solid {rcol};
                 padding:24px;text-align:center;margin:12px 0'>
              <div style='font-size:52px;font-weight:700;color:{rcol};letter-spacing:-2px'>
                {final_p*100:.1f}%</div>
              <div style='font-size:14px;color:#5C6B7A;margin:6px 0'>Combined Churn Probability</div>
              <div style='font-size:17px;font-weight:700;color:{rcol}'>{rlbl}</div>
              <div style='display:inline-block;margin-top:10px;background:{bcol};color:#FAFAF7;
                   font-size:12px;font-weight:700;letter-spacing:1px;padding:5px 18px;
                   text-transform:uppercase'>{badge}</div></div>""", unsafe_allow_html=True)

            fig_g=go.Figure(go.Indicator(mode="gauge+number",value=final_p*100,
                number={"suffix":"%","font":{"color":rcol,"size":38,"family":"Arial"}},
                gauge={"axis":{"range":[0,100],"tickcolor":GOLD},
                       "bar":{"color":rcol,"thickness":0.22},"bgcolor":"white",
                       "steps":[{"range":[0,20],"color":"#E8F5E9"},{"range":[20,40],"color":"#FFF8E1"},
                                 {"range":[40,65],"color":"#FFEBEE"},{"range":[65,100],"color":"#FFDDE1"}],
                       "threshold":{"line":{"color":GOLD,"width":3},"thickness":0.75,"value":thresh*100}}))
            fig_g.update_layout(height=220,margin=dict(t=8,b=8,l=28,r=28),
                paper_bgcolor="rgba(0,0,0,0)",font=dict(family="Times New Roman",color=NAVY))
            st.plotly_chart(fig_g, use_container_width=True)
            st.caption(f"Risk: 0-20% Low | 20-40% Moderate | 40-65% High | 65%+ Very High | Threshold: {thresh:.2f}")

            st.markdown("#### Key Risk Factors")
            pdz  = 1 if 50<=exam_pct<65 else 0
            obc  = 1 if caste=="OBC" else 0
            sci  = 1 if spec=="SCIENCE" else 0
            dh   = 1 if district in HIGH_CHURN_DISTS else 0
            cbse = 1 if "CBSE" in board.upper() else 0
            factors=[
                ("Semester",SEM_LABELS[semester].split("--")[0].strip(),
                 {1:RED,2:AMBER,3:GREEN,4:GREEN,5:GREEN,6:GREEN}[semester]),
                ("HSC %",f"{exam_pct:.1f}%", RED if pdz else AMBER if exam_pct<45 else GREEN),
                ("Zone","Danger 50-65%" if pdz else ("Very Low" if exam_pct<45 else "Safe"),
                 RED if pdz else AMBER if exam_pct<45 else GREEN),
                ("Caste",caste, AMBER if obc else GREEN),
                ("Stream",spec, RED if sci else AMBER if spec=="ARTS" else GREEN),
                ("District",district[:14], RED if dh else GREEN if district in LOCAL_DISTS else AMBER),
                ("Board",board, RED if cbse else GREEN),
                ("Religion",religion, AMBER if religion=="Muslim" else GREEN),
            ]
            fcols=st.columns(4)
            for i,(fn2,fv,fc) in enumerate(factors):
                fcols[i%4].markdown(f"""<div style='background:#FFF;border:1px solid #D4C5A9;
                     border-top:4px solid {fc};padding:10px;margin-bottom:7px;text-align:center'>
                  <div style='font-size:9px;color:#8FA3B8;font-weight:700;text-transform:uppercase'>{fn2}</div>
                  <div style='font-size:12.5px;font-weight:700;color:{fc};margin-top:4px'>{fv}</div></div>""",
                    unsafe_allow_html=True)

            recs=[]
            if semester>=4:
                recs.append(f"Semester {semester} -- student has passed the critical early semesters.")
            else:
                if semester==1: recs.append("Semester 1 -- highest risk. Immediate welfare check-in recommended.")
                if semester==2: recs.append("Semester 2 -- second highest risk. Regular monitoring advised.")
                if pdz: recs.append(f"HSC {exam_pct:.0f}% falls in the 50-65% danger zone -- highest churn bracket.")
                elif exam_pct<45: recs.append(f"HSC {exam_pct:.0f}% -- very low. Academic bridging support needed.")
                if obc: recs.append("OBC category has the highest churn rate (18.1%). Verify scholarship status.")
                if sci: recs.append("Science stream for BCA -- confirm student motivation and course fit.")
                if dh: recs.append(f"{district} is a high-risk district. Check commute and accommodation.")
                if cbse: recs.append("CBSE board shows 23.1% churn -- possible adjustment difficulty.")
                if religion=="Muslim" and district not in LOCAL_DISTS:
                    recs.append("Muslim student from non-local district -- assign peer mentor.")
                if pred and not recs:
                    recs.append(f"Combined score {final_p*100:.1f}% -- schedule a welfare check-in.")
            if recs:
                st.markdown("#### Recommended Interventions")
                for r in recs:
                    st.markdown(f"""<div style='background:#FFF;border:1px solid #D4C5A9;
                         border-left:5px solid {GOLD};padding:10px 14px;margin-bottom:7px;
                         font-size:13px;color:{NAVY}'>&nbsp; {r}</div>""", unsafe_allow_html=True)
            elif not pred:
                st.success("No major risk factors detected. Student profile appears stable.")

    with t3:
        section_header("All Test Student Predictions")
        mc1,mc2,mc3,mc4=st.columns(4)
        mc1.metric("Total Test",len(y_test))
        mc2.metric("Pred Churned",int(all_preds.sum()))
        mc3.metric("Actual Churned",int(y_test.sum()))
        mc4.metric("Correct",int((all_preds==y_test.values).sum()))
        st.divider()
        res=pd.DataFrame({
            "Semester": sem_test.values,
            "Actual":   ["Churned" if v==1 else "Active" for v in y_test.values],
            "Score %":  (all_probs*100).round(1),
            "Predicted":["Churned" if v==1 else "Active" for v in all_preds],
            "Risk":     [get_risk(p)[0] for p in all_probs],
            "Correct":  ["YES" if p==a else "NO" for p,a in zip(all_preds,y_test.values)],
        }).reset_index(drop=True)
        fc1,fc2,fc3=st.columns(3)
        fa =fc1.selectbox("Filter Actual",["All","Churned","Active"])
        fp2=fc2.selectbox("Filter Predicted",["All","Churned","Active"])
        fcr=fc3.selectbox("Filter Correct",["All","YES","NO"])
        flt=res.copy()
        if fa !="All": flt=flt[flt["Actual"]==fa]
        if fp2!="All": flt=flt[flt["Predicted"]==fp2]
        if fcr!="All": flt=flt[flt["Correct"]==fcr]
        st.markdown(f"**Showing {len(flt)} of {len(res)} students**")
        st.dataframe(flt, use_container_width=True, height=360)
        fig_h=px.histogram(res,x="Score %",color="Actual",nbins=30,barmode="overlay",opacity=0.75,
            color_discrete_map={"Active":GREEN,"Churned":RED},
            title="Combined score distribution -- all test students")
        fig_h.add_vline(x=thresh*100,line_dash="dash",line_color=GOLD,
            annotation_text=f"Threshold {thresh:.2f}",annotation_font_color=GOLD,
            annotation_font_size=13)
        tnr(fig_h,360); st.plotly_chart(fig_h, use_container_width=True)
