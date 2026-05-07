import streamlit as st
from main import analyse
from src.features import extract_features, get_feature_vector, is_whitelisted
from src.explainer import get_risk_score, get_explanation
from src.ml_models import load_model
from src.rule_based import predict as rule_predict
from config import MODEL_DIR, LABEL_NAMES
import os
from pathlib import Path
import json

def load_threshold(model_dir: str) -> float:
    try:
        path = os.path.join(model_dir, "threshold.json")
        with open(path, "r") as f:
            data = json.load(f)
        return float(data["threshold"])
    except Exception:
        return 0.55  # fallback default

# Get absolute path to models directory
PROJECT_ROOT = Path(__file__).parent.absolute()
MODELS_DIR = PROJECT_ROOT / "models"

# Page config
st.set_page_config(
    page_title="PhishGuard",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── GLOBAL RESET ── */
    html, body, [class*="css"], [data-testid="stAppViewContainer"],
    [data-testid="stApp"], .main, section.main {
        background-color: #F4F2FF !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        color: #1E1B4B !important;
    }

    /* Hide Streamlit chrome */
    #MainMenu {display: none !important;}
    header[data-testid="stHeader"] {display: none !important;}
    footer {display: none !important;}
    .viewerBadge_container__1QSob {display: none !important;}
    [data-testid="stToolbar"] {display: none !important;}
    [data-testid="stDecoration"] {display: none !important;}

    /* ── BLOCK CONTAINER ── */
    .block-container {
        padding-top: 0 !important;
        padding-bottom: 2rem !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
        max-width: 100% !important;
    }

    /* ── TOP BAR ── */
    .top-bar {
        background: linear-gradient(135deg, #6D5FD5 0%, #9B8AFB 100%);
        padding: 16px 32px;
        margin-bottom: 28px;
        margin-left: -2.5rem;
        margin-right: -2.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 24px rgba(109, 95, 213, 0.25);
    }

    .top-bar-left {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 20px;
        font-weight: 700;
        color: #FFFFFF !important;
        letter-spacing: -0.3px;
    }

    .top-bar-right {
        font-size: 12px;
        color: rgba(255, 255, 255, 0.80) !important;
        font-weight: 500;
        letter-spacing: 0.2px;
    }

    /* ── LABELS ── */
    .label-small {
        font-size: 13px;
        color: #9B8AFB !important;
        font-weight: 700;
        margin-bottom: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* ── CARDS ── */
    .card {
        background-color: #FFFFFF !important;
        border: 1px solid #E5E0FF;
        border-radius: 14px;
        padding: 22px 24px;
        margin-bottom: 16px;
        box-shadow: 0 2px 16px rgba(109, 95, 213, 0.07);
        transition: box-shadow 0.25s ease, transform 0.25s ease;
    }

    .card:hover {
        box-shadow: 0 4px 28px rgba(109, 95, 213, 0.13);
        transform: translateY(-1px);
    }

    /* ── VERDICT BANNERS ── */
    .verdict-phishing {
        background: linear-gradient(135deg, #FFF1F1 0%, #FFF5F5 100%) !important;
        border-left: 4px solid #F87171 !important;
    }

    .verdict-legitimate {
        background: linear-gradient(135deg, #F0FFF9 0%, #F5FFFB 100%) !important;
        border-left: 4px solid #34D399 !important;
    }

    .verdict-text {
        font-size: 24px;
        font-weight: 700;
        margin-bottom: 12px;
        letter-spacing: -0.3px;
    }

    .verdict-phishing-text {
        color: #DC4A4A !important;
    }

    .verdict-legitimate-text {
        color: #059669 !important;
    }

    /* ── PILLS ── */
    .pill {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        margin-right: 8px;
        margin-top: 6px;
        letter-spacing: 0.3px;
    }

    .pill-phishing {
        background-color: #FEE2E2;
        color: #DC4A4A;
        border: 1px solid #FECACA;
    }

    .pill-legitimate {
        background-color: #D1FAE5;
        color: #059669;
        border: 1px solid #A7F3D0;
    }

    /* ── RISK SCORE ── */
    .risk-score-container {
        text-align: center;
        padding: 8px 0;
    }

    .risk-score-number {
        font-size: 56px;
        font-weight: 700;
        margin-bottom: 4px;
        letter-spacing: -2px;
        line-height: 1;
    }

    .risk-score-label {
        font-size: 15px;
        color: #8B85C1 !important;
        margin-bottom: 4px;
    }

    .risk-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 8px;
        font-size: 11px;
        font-weight: 700;
        margin-top: 14px;
        letter-spacing: 0.5px;
    }

    .risk-badge-high {
        background-color: #FEE2E2;
        color: #DC4A4A;
    }

    .risk-badge-suspicious {
        background-color: #FEF3C7;
        color: #D97706;
    }

    .risk-badge-low {
        background-color: #D1FAE5;
        color: #059669;
    }

    /* ── FEATURE ROWS ── */
    .feature-row {
        padding: 13px 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #F3F0FF;
    }

    .feature-row:last-child {
        border-bottom: none;
    }

    .feature-label {
        color: #8B85C1 !important;
        font-size: 15px;
        font-weight: 500;
    }

    .feature-value {
        color: #1E1B4B !important;
        font-size: 15px;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* ── REASON ITEMS ── */
    .reason-item {
        padding: 10px 0;
        color: #1E1B4B !important;
        font-size: 15px;
        border-bottom: 1px solid #F3F0FF;
        line-height: 1.5;
    }

    .reason-item:last-child {
        border-bottom: none;
    }

    .reason-dot {
        color: #F87171;
        margin-right: 8px;
        font-size: 16px;
    }

    .reason-dot-safe {
        color: #34D399;
    }

    /* ── COMPARISON ── */
    .comparison-row {
        display: flex;
        gap: 16px;
        margin-bottom: 12px;
    }

    .comparison-col {
        flex: 1;
        text-align: center;
    }

    .comparison-label {
        font-size: 13px;
        color: #8B85C1 !important;
        margin-bottom: 10px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .comparison-value {
        font-size: 15px;
        font-weight: 700;
        padding: 10px 16px;
        border-radius: 8px;
        display: inline-block;
    }

    .comparison-phishing {
        background-color: #FEE2E2;
        color: #DC4A4A;
    }

    .comparison-legitimate {
        background-color: #D1FAE5;
        color: #059669;
    }

    /* ── DIVIDER ── */
    hr {
        border: none !important;
        border-top: 1px solid #EDE9FE !important;
        margin: 20px 0 !important;
    }

    /* ── HELPER TEXT ── */
    .helper-text {
        font-size: 13px;
        color: #8B85C1 !important;
        margin-top: 8px;
    }

    /* ── STREAMLIT INPUT ── */
    .stTextInput > div > div > input {
        background-color: #FAFAFE !important;
        border: 1.5px solid #E5E0FF !important;
        border-radius: 10px !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 15px !important;
        color: #1E1B4B !important;
        padding: 14px 16px !important;
        transition: all 0.2s ease !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: #7C6FF7 !important;
        box-shadow: 0 0 0 3px rgba(124, 111, 247, 0.15) !important;
        background-color: #FFFFFF !important;
    }

    .stTextInput > div > div > input::placeholder {
        color: #C4BFEA !important;
    }

    /* ── STREAMLIT BUTTONS ── */
    .stButton > button {
        background: linear-gradient(135deg, #7C6FF7 0%, #9B8AFB 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 14px 28px !important;
        font-weight: 600 !important;
        font-size: 16px !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        box-shadow: 0 4px 14px rgba(124, 111, 247, 0.35) !important;
        transition: all 0.2s ease !important;
        letter-spacing: 0.1px !important;
    }

    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 22px rgba(124, 111, 247, 0.45) !important;
        filter: brightness(1.05) !important;
    }

    .stButton > button:active {
        transform: translateY(0px) !important;
        filter: brightness(0.98) !important;
    }

    /* ── PROGRESS BAR ── */
    [data-testid="stProgressBar"] {
        background-color: #EDE9FE !important;
        border-radius: 6px !important;
        height: 6px !important;
    }

    [data-testid="stProgressBar"] > div {
        background: linear-gradient(90deg, #7C6FF7, #9B8AFB) !important;
        border-radius: 6px !important;
    }

    /* Also target inner progress fill */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #7C6FF7, #9B8AFB) !important;
        border-radius: 6px !important;
    }

    /* ── ALERTS ── */
    [data-testid="stAlert"] {
        border-radius: 10px !important;
        border: 1px solid #E5E0FF !important;
        background-color: #FAFAFE !important;
    }

    /* ── SPINNER ── */
    [data-testid="stSpinner"] {
        color: #7C6FF7 !important;
    }

    /* ── STREAMLIT COLUMNS gap ── */
    [data-testid="column"] {
        background: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'show_results' not in st.session_state:
    st.session_state.show_results = False
if 'ml_pred' not in st.session_state:
    st.session_state.ml_pred = None
if 'rule_pred' not in st.session_state:
    st.session_state.rule_pred = None
if 'risk_score' not in st.session_state:
    st.session_state.risk_score = None
if 'risk_level' not in st.session_state:
    st.session_state.risk_level = None
if 'reasons' not in st.session_state:
    st.session_state.reasons = None
if 'feature_dict' not in st.session_state:
    st.session_state.feature_dict = None
if 'url' not in st.session_state:
    st.session_state.url = ""

# Top bar
st.markdown("""
<div class="top-bar">
    <div class="top-bar-left">🛡️ PhishGuard</div>
    <div class="top-bar-right">Powered by Random Forest + Rule Engine</div>
</div>
""", unsafe_allow_html=True)

# Main content - use columns to limit width
col1, col2, col3 = st.columns([1, 3, 1])

with col2:
    # Input section
    st.markdown('<div class="label-small">Enter URL to analyse</div>', unsafe_allow_html=True)
    url_input = st.text_input(
        "URL input",
        label_visibility="collapsed",
        placeholder="https://example.com",
        key="url_field"
    )

    # Analyze button
    if st.button("Analyse", use_container_width=True, key="analyze_btn"):
        if not url_input.strip():
            st.warning("⚠ Please enter a URL to analyse")
        else:
            try:
                with st.spinner("🔍 Analysing URL..."):
                    from src.features import is_whitelisted, extract_features, get_feature_vector
                    from src.explainer import get_risk_score, get_explanation
                    from src.rule_based import predict as rule_predict
                    from src.ml_models import load_model
                    from config import MODEL_DIR

                    model = load_model("best_model", MODEL_DIR)

                    if is_whitelisted(url_input):
                        st.session_state.ml_pred = 0
                        st.session_state.rule_pred = 0
                        st.session_state.risk_score = 2.0
                        st.session_state.risk_level = "LOW RISK"
                        st.session_state.reasons = ["Domain found in Tranco Top 1M trusted list"]
                        st.session_state.feature_dict = extract_features(url_input)
                    else:
                        feature_dict = extract_features(url_input)
                        feature_vector = get_feature_vector(url_input)
                        proba = float(model.predict_proba(feature_vector)[0][1])
                        threshold = load_threshold(MODEL_DIR)
                        ml_pred = 1 if proba >= threshold else 0
                        rule_pred = rule_predict(feature_dict, url_input)
                        risk_score, risk_level = get_risk_score(model, feature_vector, feature_dict, url_input)
                        reasons = get_explanation(url_input, ml_pred, feature_dict)
                        st.session_state.ml_pred = ml_pred
                        st.session_state.rule_pred = rule_pred
                        st.session_state.risk_score = risk_score
                        st.session_state.risk_level = risk_level
                        st.session_state.reasons = reasons
                        st.session_state.feature_dict = feature_dict

                    st.session_state.url = url_input
                    st.session_state.show_results = True
            except Exception as e:
                st.error(f"❌ Error analysing URL: {str(e)}")

    st.markdown('<div class="helper-text">Supports any URL format — http, https, IP-based</div>', unsafe_allow_html=True)

    # Results section
    if st.session_state.show_results:
        ml_pred = st.session_state.ml_pred
        rule_pred = st.session_state.rule_pred
        risk_score = st.session_state.risk_score
        risk_level = st.session_state.risk_level
        reasons = st.session_state.reasons
        feature_dict = st.session_state.feature_dict
        url = st.session_state.url

        st.markdown("<hr>", unsafe_allow_html=True)

        # Verdict Banner
        ml_label = LABEL_NAMES.get(ml_pred, ['Legitimate', 'Phishing'][ml_pred])
        rule_label = LABEL_NAMES.get(rule_pred, ['Legitimate', 'Phishing'][rule_pred])

        # Combined verdict: if either classifier says phishing, flag it
        final_pred = 1 if (ml_pred == 1 or rule_pred == 1) else 0

        if final_pred == 1:  # Phishing
            verdict_icon = "⚠️"
            verdict_text = "Phishing Detected"
            verdict_class = "verdict-phishing"
            verdict_text_class = "verdict-phishing-text"
            pill_class = "pill-phishing"
        else:  # Legitimate
            verdict_icon = "✅"
            verdict_text = "Legitimate"
            verdict_class = "verdict-legitimate"
            verdict_text_class = "verdict-legitimate-text"
            pill_class = "pill-legitimate"

        st.markdown(f"""
        <div class="card {verdict_class}">
            <div class="verdict-text {verdict_text_class}">{verdict_icon} {verdict_text}</div>
            <div>
                <span class="pill {pill_class}">Rule-Based: {rule_label}</span>
                <span class="pill {pill_class}">ML Model: {ml_label}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Two-column layout for Risk Score and Reasons
        col_risk, col_reasons = st.columns(2)

        with col_risk:
            risk_color = "#F87171" if risk_level == "HIGH" else \
                        "#FBBF24" if risk_level == "SUSPICIOUS" else "#34D399"
            risk_badge_class = "risk-badge-high" if risk_level == "HIGH" else \
                              "risk-badge-suspicious" if risk_level == "SUSPICIOUS" else "risk-badge-low"
            progress_pct = min(risk_score, 100)

            st.markdown(f"""
            <div class="card">
                <div class="label-small">Risk Score</div>
                <div class="risk-score-container">
                    <div class="risk-score-number" style="color:{risk_color};">{risk_score:.1f}</div>
                    <div class="risk-score-label">/100</div>
                    <div class="risk-badge {risk_badge_class}">{risk_level} RISK</div>
                    <div style="background:#EDE9FE; border-radius:6px; height:8px; margin:18px 0 0 0;">
                        <div style="background:linear-gradient(90deg,#7C6FF7,#9B8AFB); width:{progress_pct:.1f}%; height:100%; border-radius:6px; transition:width 0.4s ease;"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_reasons:
            # Detection Reasons Card
            reasons_items = ""
            if reasons:
                for reason in reasons:
                    reasons_items += f'<div class="reason-item"><span class="reason-dot">•</span>{reason}</div>'
            else:
                reasons_items = '<div class="reason-item"><span class="reason-dot reason-dot-safe">✓</span>No suspicious patterns detected</div>'
 
            st.markdown(f"""
            <div class="card" style="height: 100%;">
                <div class="label-small">Detection Reasons</div>
                {reasons_items}
            </div>
            """, unsafe_allow_html=True)

        # Feature Breakdown Card
        url_length    = feature_dict.get('url_length', 0)
        length_pct    = min((url_length / 150) * 100, 100)
        has_https     = feature_dict.get('has_https', False)
        https_text    = "Yes" if has_https else "No"
        https_color   = "#34D399" if has_https else "#F87171"
        entropy       = feature_dict.get('shannon_entropy', 0)
        entropy_desc  = "Low" if entropy < 3.5 else ("Medium" if entropy < 4.0 else "High — suspicious")
        has_ip        = feature_dict.get('has_ip_address', False)
        ip_text       = "Yes" if has_ip else "No"
        ip_color      = "#F87171" if has_ip else "#34D399"
        special_chars = feature_dict.get('special_char_count', 0)
        subdomain_depth = feature_dict.get('subdomain_depth', 0)

        st.markdown(f"""<div class="card"><div class="label-small">Feature Analysis</div><div style="display:grid; grid-template-columns:1fr 1fr; gap:0 40px;"><div><div class="feature-row"><div class="feature-label">URL Length</div><div class="feature-value">{url_length}</div></div><div style="background:#EDE9FE; border-radius:6px; height:6px; margin:4px 0 12px 0;"><div style="background:linear-gradient(90deg,#7C6FF7,#9B8AFB); width:{length_pct:.1f}%; height:100%; border-radius:6px;"></div></div><div class="feature-row"><div class="feature-label">HTTPS Protocol</div><div class="feature-value" style="color:{https_color};">{https_text}</div></div><div class="feature-row"><div class="feature-label">Shannon Entropy</div><div class="feature-value">{entropy:.2f} ({entropy_desc})</div></div></div><div><div class="feature-row"><div class="feature-label">IP Address</div><div class="feature-value" style="color:{ip_color};">{ip_text}</div></div><div class="feature-row"><div class="feature-label">Special Characters</div><div class="feature-value">{special_chars}</div></div><div class="feature-row"><div class="feature-label">Subdomain Depth</div><div class="feature-value">{subdomain_depth}</div></div></div></div></div>""", unsafe_allow_html=True)

        # Verdict Comparison
        st.markdown("""
        <div class="card">
            <div class="label-small">Verdicts Comparison</div>
            <div class="comparison-row">
                <div class="comparison-col">
                    <div class="comparison-label">Rule-Based Engine</div>
                    <div class="comparison-value comparison-""" +
            ("phishing" if rule_pred == 1 else "legitimate") +
            f"""\">{rule_label}</div>
                </div>
                <div class="comparison-col">
                    <div class="comparison-label">ML Model (Random Forest)</div>
                    <div class="comparison-value comparison-""" +
            ("phishing" if ml_pred == 1 else "legitimate") +
            f"""\">{ml_label}</div>
                </div>
            </div>
            <div style="font-size: 12px; color: #8B85C1; text-align: center; margin-top: 14px;">
                Discrepancies between classifiers may indicate borderline URLs
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Clear results button
        if st.button("↻ Analyse another URL", use_container_width=False, key="clear_btn"):
            st.session_state.show_results = False
            st.session_state.ml_pred = None
            st.session_state.rule_pred = None
            st.session_state.risk_score = None
            st.session_state.risk_level = None
            st.session_state.reasons = None
            st.session_state.feature_dict = None
            st.session_state.url = ""
            st.rerun()