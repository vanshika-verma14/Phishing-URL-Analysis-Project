"""
Live predictor for the phishing-detector project.

This script loads the trained model and analyzes any URL for phishing indicators.
Run this after train.py has successfully completed.
"""

import sys
import json
import os

def load_threshold(model_dir: str) -> float:
    try:
        path = os.path.join(model_dir, "threshold.json")
        with open(path, "r") as f:
            data = json.load(f)
        return float(data["threshold"])
    except Exception:
        return 0.55  # fallback default


def analyse(url: str) -> int:
    """
    Analyze a URL for phishing indicators.
    
    Loads the trained model and runs the full detection pipeline including
    rule-based classification, ML prediction, risk scoring, and explainability.
    Prints a comprehensive report and returns the ML prediction.
    
    Args:
        url (str): The URL to analyze
        
    Returns:
        int: ML model prediction (0 for legitimate, 1 for phishing)
    """
    from src.features import is_whitelisted
    
    if is_whitelisted(url):
        print("=" * 44)
        print("  PHISHING DETECTION REPORT")
        print("=" * 44)
        print(f"  URL      : {url}")
        print("  " + "-" * 42)
        print("  Verdict  : ✓ LEGITIMATE")
        print("  Reason   : Domain verified in Tranco Top")
        print("             1 Million trusted domains list")
        print("  Risk Score: 2.0/100 — LOW RISK")
        print("=" * 44)
        return 0
    
    try:
        from src.ml_models import load_model
        from src.features import extract_features, get_feature_vector
        from src.rule_based import predict as rule_predict
        from src.explainer import get_explanation, get_risk_score, print_explanation
        from config import MODEL_DIR
        
        # Load model
        model = load_model("best_model", MODEL_DIR)
        
        # Extract features
        feature_vector = get_feature_vector(url)
        feature_dict = extract_features(url)
        
        # Get predictions
        proba = float(model.predict_proba(feature_vector)[0][1])
        threshold = load_threshold(MODEL_DIR)
        ml_pred = 1 if proba >= threshold else 0
        rule_pred = rule_predict(feature_dict, url)
        
        # Combined verdict: if either classifier says phishing, flag it
        final_pred = 1 if (ml_pred == 1 or rule_pred == 1) else 0
        
        # Get risk score
        risk_score, risk_level = get_risk_score(model, feature_vector, feature_dict, url)
        
        # Get explanation
        reasons = get_explanation(url, final_pred, feature_dict)
        
        # Print report
        print_explanation(url, final_pred, rule_pred, feature_dict, reasons, risk_score, risk_level)
        
        return int(final_pred)
    
    except FileNotFoundError:
        print("\n" + "="*60)
        print("ERROR")
        print("="*60)
        print("\nModel not found. Please run train.py first.\n")
        sys.exit(1)


def main():
    """
    Run the live analyser with demo URLs.
    
    Prints startup banner and demonstrates the phishing detector on
    a selection of example URLs (both legitimate and phishing patterns).
    """
    print("="*60)
    print("   PHISHING DETECTOR — LIVE ANALYSER")
    print("   Model: Random Forest | Run train.py to retrain")
    print("="*60)
    
    # Demo URLs
    demo_urls = [
        "https://www.google.com",
        "https://github.com/login",
        "http://192.168.1.1/secure/login/verify?user=admin&pass=1234",
        "http://paypa1-secure-login.xyz/account/confirm?id=8823&token=abc",
        "http://free-iphone-winner.tk/claim?ref=abc123&session=xyz987"
    ]
    
    for i, url in enumerate(demo_urls, 1):
        print(f"\n{'='*60}")
        print(f"DEMO {i}/5")
        print(f"{'='*60}")
        analyse(url)
    
    print("="*60)
    print("To analyse a custom URL, call: analyse('your-url-here')")
    print("="*60)


if __name__ == "__main__":
    main()
