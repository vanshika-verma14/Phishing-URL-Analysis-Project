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
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;700&display=swap');
    
    * {
        font-family: 'DM Sans', sans-serif;
    }
    
    /* Hide Streamlit defaults */
    #MainMenu {display: none;}
    header {display: none;}
    footer {display: none;}
    .viewerBadge_container__1QSob {display: none;}
    
    /* Main background */
    body {
        background-color: #F7F8FA;
    }
    
    .main {
        background-color: #F7F8FA;
        padding: 0;
    }
    
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    
    /* Text colors */
    .primary-text {
        color: #111827;
    }
    
    .secondary-text {
        color: #6B7280;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: white;
        color: #111827;
        border: 1px solid #E5E7EB;
        border-radius: 6px;
        padding: 10px 16px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background-color: #F3F4F6;
        border-color: #D1D5DB;
    }
    
    /* Input fields */
    input {
        font-family: 'DM Mono', monospace !important;
    }
    
    .stTextInput > div > div > input {
        border: 1px solid #E5E7EB !important;
        border-radius: 6px !important;
        font-family: 'DM Mono', monospace !important;
    }
    
    /* Cards */
    .card {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    
    /* Top bar */
    .top-bar {
        background-color: #FFFFFF;
        border-bottom: 1px solid #E5E7EB;
        padding: 12px 20px;
        margin-bottom: 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    
    .top-bar-left {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 18px;
        font-weight: 700;
        color: #111827;
    }
    
    .top-bar-right {
        font-size: 12px;
        color: #6B7280;
    }
    
    /* Labels */
    .label-small {
        font-size: 12px;
        color: #6B7280;
        font-weight: 500;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Verdict banner */
    .verdict-banner {
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 16px;
        border-left: 4px solid;
    }
    
    .verdict-phishing {
        background-color: #FEF2F2;
        border-left-color: #DC2626;
    }
    
    .verdict-legitimate {
        background-color: #F0FDF4;
        border-left-color: #16A34A;
    }
    
    .verdict-text {
        font-size: 20px;
        font-weight: 700;
        margin-bottom: 12px;
    }
    
    .verdict-phishing-text {
        color: #DC2626;
    }
    
    .verdict-legitimate-text {
        color: #16A34A;
    }
    
    /* Pills */
    .pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 11px;
        font-weight: 600;
        margin-right: 8px;
        margin-top: 8px;
    }
    
    .pill-phishing {
        background-color: #FEE2E2;
        color: #DC2626;
    }
    
    .pill-legitimate {
        background-color: #DCFCE7;
        color: #16A34A;
    }
    
    /* Risk score */
    .risk-score-container {
        text-align: center;
    }
    
    .risk-score-number {
        font-size: 48px;
        font-weight: 700;
        margin-bottom: 8px;
    }
    
    .risk-score-label {
        font-size: 12px;
        color: #6B7280;
    }
    
    .risk-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 600;
        margin-top: 12px;
    }
    
    .risk-badge-high {
        background-color: #FEE2E2;
        color: #DC2626;
    }
    
    .risk-badge-suspicious {
        background-color: #FEF3C7;
        color: #D97706;
    }
    
    .risk-badge-low {
        background-color: #DCFCE7;
        color: #16A34A;
    }
    
    /* Feature row */
    .feature-row {
        padding: 12px 0;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid #F3F4F6;
    }
    
    .feature-row:last-child {
        border-bottom: none;
    }
    
    .feature-label {
        color: #6B7280;
        font-size: 14px;
    }
    
    .feature-value {
        color: #111827;
        font-size: 14px;
        font-weight: 500;
        font-family: 'DM Mono', monospace;
    }
    
    /* Reason item */
    .reason-item {
        padding: 12px 0;
        color: #111827;
        font-size: 14px;
        border-bottom: 1px solid #F3F4F6;
    }
    
    .reason-item:last-child {
        border-bottom: none;
    }
    
    .reason-dot {
        color: #DC2626;
        margin-right: 8px;
    }
    
    .reason-dot-safe {
        color: #16A34A;
    }
    
    /* Comparison table */
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
        font-size: 12px;
        color: #6B7280;
        margin-bottom: 8px;
        font-weight: 500;
    }
    
    .comparison-value {
        font-size: 14px;
        font-weight: 600;
        padding: 8px 12px;
        border-radius: 6px;
        display: inline-block;
    }
    
    .comparison-phishing {
        background-color: #FEE2E2;
        color: #DC2626;
    }
    
    .comparison-legitimate {
        background-color: #DCFCE7;
        color: #16A34A;
    }
    
    /* Divider */
    hr {
        border: none;
        border-top: 1px solid #F3F4F6;
        margin: 16px 0;
    }
    
    /* Warning/Error styling */
    .stWarning, .stError, .stSuccess {
        border-radius: 8px;
    }
    
    /* Helper text */
    .helper-text {
        font-size: 12px;
        color: #6B7280;
        margin-top: 8px;
    }
    
    /* Clear button */
    .clear-link {
        color: #2563EB;
        text-decoration: none;
        font-size: 14px;
        cursor: pointer;
    }
    
    .clear-link:hover {
        text-decoration: underline;
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
    <div class="top-bar-left">🛡 PhishGuard</div>
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
            verdict_icon = "⚠"
            verdict_text = "Phishing Detected"
            verdict_class = "verdict-phishing"
            verdict_text_class = "verdict-phishing-text"
            pill_class = "pill-phishing"
        else:  # Legitimate
            verdict_icon = "✓"
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
            # Risk Score Card
            risk_color = "#DC2626" if risk_level == "HIGH" else \
                        "#D97706" if risk_level == "SUSPICIOUS" else "#16A34A"
            
            risk_badge_class = "risk-badge-high" if risk_level == "HIGH" else \
                              "risk-badge-suspicious" if risk_level == "SUSPICIOUS" else "risk-badge-low"
            
            st.markdown(f"""
            <div class="card">
                <div class="label-small">Risk Score</div>
                <div class="risk-score-container">
                    <div class="risk-score-number" style="color: {risk_color};">{risk_score:.1f}</div>
                    <div class="risk-score-label">/100</div>
                    <div class="risk-badge {risk_badge_class}">{risk_level} RISK</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Add progress bar
            progress_value = min(risk_score / 100, 1.0)
            st.progress(progress_value, text=None)
        
        with col_reasons:
            # Detection Reasons Card
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="label-small">Detection Reasons</div>', unsafe_allow_html=True)
            
            if reasons:
                for reason in reasons:
                    st.markdown(f'<div class="reason-item"><span class="reason-dot">•</span>{reason}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="reason-item"><span class="reason-dot reason-dot-safe">✓</span>No suspicious patterns detected</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Feature Breakdown Card
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="label-small">Feature Analysis</div>', unsafe_allow_html=True)
        
        col_f1, col_f2 = st.columns(2)
        
        with col_f1:
            # URL Length
            url_length = feature_dict.get('url_length', 0)
            length_pct = min((url_length / 150) * 100, 100)
            st.markdown(f"""
            <div class="feature-row">
                <div class="feature-label">URL Length</div>
                <div class="feature-value">{url_length}</div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(length_pct / 100, text=None)
            
            # HTTPS
            has_https = feature_dict.get('has_https', False)
            https_text = "Yes" if has_https else "No"
            https_color = "#16A34A" if has_https else "#DC2626"
            st.markdown(f"""
            <div class="feature-row">
                <div class="feature-label">HTTPS Protocol</div>
                <div class="feature-value" style="color: {https_color};">{https_text}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Shannon Entropy
            entropy = feature_dict.get('shannon_entropy', 0)
            if entropy < 3.5:
                entropy_desc = "Low"
            elif entropy < 4.0:
                entropy_desc = "Medium"
            else:
                entropy_desc = "High — suspicious"
            st.markdown(f"""
            <div class="feature-row">
                <div class="feature-label">Shannon Entropy</div>
                <div class="feature-value">{entropy:.2f} ({entropy_desc})</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col_f2:
            # Has IP Address
            has_ip = feature_dict.get('has_ip_address', False)
            ip_text = "Yes" if has_ip else "No"
            ip_color = "#DC2626" if has_ip else "#16A34A"
            st.markdown(f"""
            <div class="feature-row">
                <div class="feature-label">IP Address</div>
                <div class="feature-value" style="color: {ip_color};">{ip_text}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Special Characters
            special_chars = feature_dict.get('special_char_count', 0)
            st.markdown(f"""
            <div class="feature-row">
                <div class="feature-label">Special Characters</div>
                <div class="feature-value">{special_chars}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Subdomain Depth
            subdomain_depth = feature_dict.get('subdomain_depth', 0)
            st.markdown(f"""
            <div class="feature-row">
                <div class="feature-label">Subdomain Depth</div>
                <div class="feature-value">{subdomain_depth}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
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
            <div style="font-size: 12px; color: #6B7280; text-align: center; margin-top: 12px;">
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
