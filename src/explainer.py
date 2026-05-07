"""
Explainability and sample predictions module for phishing detection.

This module generates human-readable explanations for predictions and produces
mandatory sample prediction tables for project evaluation.
"""

import pandas as pd

from config import (
    THRESHOLD_URL_LENGTH, THRESHOLD_SPECIAL_CHARS, THRESHOLD_DOT_COUNT,
    THRESHOLD_ENTROPY, THRESHOLD_SUBDOMAIN_DEPTH,
    RISK_LOW_MAX, RISK_SUSPICIOUS_MAX, LABEL_NAMES
)
from src.features import extract_features, FEATURE_COLUMNS


def get_explanation(url: str, prediction: int, feature_dict: dict) -> list:
    """
    Generate human-readable reasons for a phishing detection decision.
    
    Analyzes feature values against thresholds to build a list of explanations
    for why a URL was flagged or not flagged as phishing.
    
    Args:
        url (str): The original URL string
        prediction (int): ML model prediction (0 for legitimate, 1 for phishing)
        feature_dict (dict): Feature dictionary from extract_features()
        
    Returns:
        list: List of reason strings explaining the prediction
    """
    reasons = []
    
    # Check each feature against thresholds
    if feature_dict["url_length"] > THRESHOLD_URL_LENGTH:
        reasons.append(f"URL is unusually long ({feature_dict['url_length']} chars)")
    
    if feature_dict["has_https"] == 0:
        reasons.append("No HTTPS — connection is not secure")
    
    if feature_dict["special_char_count"] > THRESHOLD_SPECIAL_CHARS:
        reasons.append(f"High special character count ({feature_dict['special_char_count']})")
    
    if feature_dict["has_ip_address"] == 1:
        reasons.append("URL contains raw IP address instead of domain name")
    
    if feature_dict["dot_count"] > THRESHOLD_DOT_COUNT:
        reasons.append(f"Excessive dots suggest deep subdomain nesting ({feature_dict['dot_count']} dots)")
    
    if feature_dict["shannon_entropy"] > THRESHOLD_ENTROPY:
        reasons.append(f"High entropy ({feature_dict['shannon_entropy']:.2f}) — suggests randomly generated domain")
    
    if feature_dict["subdomain_depth"] > THRESHOLD_SUBDOMAIN_DEPTH:
        reasons.append(f"Deep subdomain structure ({feature_dict['subdomain_depth']} levels)")
    
    # Handle cases with no reasons
    if len(reasons) == 0:
        if prediction == 0:
            reasons.append("No suspicious patterns detected")
        else:
            reasons.append("ML model flagged this URL based on combined feature patterns")
    
    return reasons


def get_risk_score(model, feature_vector, feature_dict, url="") -> tuple[float, str]:
    """
    Calculate risk score using weighted multi-signal scoring system.
    
    Combines multiple security signals (ML probability, IP presence, HTTPS,
    suspicious TLD, entropy, special characters) with weighted scoring to
    produce a more robust 0-100 risk score with categorical risk level.
    
    Args:
        model: Fitted ML model with predict_proba() method (e.g., Random Forest)
        feature_vector: Feature vector for ML prediction
        feature_dict (dict): Feature dictionary from extract_features()
        url (str): Original URL string for TLD analysis (optional)
        
    Returns:
        tuple: (risk_score, risk_level) where risk_score is float (0-100)
               and risk_level is str ("LOW RISK", "SUSPICIOUS", or "HIGH RISK")
    """
    from urllib.parse import urlparse
    from config import SUSPICIOUS_TLDS
    
    score = 0.0
    
    # Signal 1 — ML probability (weight: 40 points max)
    proba = float(model.predict_proba(feature_vector)[0][1])
    score += proba * 40
    
    # Signal 2 — IP address present (weight: 20 points)
    if feature_dict.get("has_ip_address", 0) == 1:
        score += 20
    
    # Signal 3 — No HTTPS (weight: 15 points)
    if feature_dict.get("has_https", 0) == 0:
        score += 15
    
    # Signal 4 — Suspicious TLD (weight: 15 points)
    try:
        netloc = urlparse(url).netloc.lower()
        for tld in SUSPICIOUS_TLDS:
            if netloc.endswith(tld):
                score += 15
                break
    except:
        pass
    
    # Signal 5 — High entropy (weight: 5 points)
    if feature_dict.get("shannon_entropy", 0) > 4.0:
        score += 5
    
    # Signal 6 — Special chars high (weight: 5 points)
    if feature_dict.get("special_char_count", 0) > 4:
        score += 5
    
    # Cap at 100
    score = min(round(score, 1), 100.0)
    
    # Risk level
    if score <= 25:
        risk_level = "LOW RISK"
    elif score <= 55:
        risk_level = "SUSPICIOUS"
    else:
        risk_level = "HIGH RISK"
    
    return score, risk_level


def print_explanation(url: str, ml_pred: int, rule_pred: int,
                      feature_dict: dict, reasons: list,
                      risk_score: float, risk_level: str) -> None:
    """
    Print a formatted phishing detection report.
    
    Generates a comprehensive report including verdicts from multiple classifiers,
    risk score, reasoning, and detailed feature breakdown.
    
    Args:
        url (str): The URL being analyzed
        ml_pred (int): ML model prediction (0 or 1)
        rule_pred (int): Rule-based model prediction (0 or 1)
        feature_dict (dict): Feature dictionary from extract_features()
        reasons (list): List of reason strings from get_explanation()
        risk_score (float): Risk score from get_risk_score()
        risk_level (str): Risk level from get_risk_score()
    """
    ml_verdict = LABEL_NAMES[ml_pred]
    rule_verdict = LABEL_NAMES[rule_pred]
    
    print("\n" + "="*50)
    print("PHISHING DETECTION REPORT")
    print("="*50)
    print(f"URL      : {url}")
    print("-"*50)
    print(f"Rule-Based Verdict : {rule_verdict}")
    print(f"ML Verdict         : {ml_verdict}")
    print(f"Risk Score         : {risk_score:.1f}/100 — {risk_level}")
    print("-"*50)
    print("Why flagged:")
    for reason in reasons:
        print(f"  - {reason}")
    print("-"*50)
    print("Feature Breakdown:")
    print(f"  URL Length       : {feature_dict['url_length']}")
    https_str = "Yes" if feature_dict['has_https'] == 1 else "No"
    print(f"  HTTPS Present    : {https_str}")
    print(f"  Special Chars    : {feature_dict['special_char_count']}")
    print(f"  Dot Count        : {feature_dict['dot_count']}")
    ip_str = "Yes" if feature_dict['has_ip_address'] == 1 else "No"
    print(f"  Has IP Address   : {ip_str}")
    print(f"  Digit Count      : {feature_dict['digit_count']}")
    print(f"  Subdomain Depth  : {feature_dict['subdomain_depth']}")
    print(f"  Shannon Entropy  : {feature_dict['shannon_entropy']:.2f}")
    print("="*50 + "\n")


def sample_predictions_table(model, X_test: pd.DataFrame,
                             y_test: pd.Series, df: pd.DataFrame) -> None:
    """
    Generate and print a sample predictions table.
    
    This is a mandatory project output. Selects 8 phishing and 7 legitimate
    samples, makes predictions, and displays them in a formatted table with
    correctness indicators.
    
    Args:
        model: Fitted ML model with predict() method
        X_test (pd.DataFrame): Feature matrix with test data
        y_test (pd.Series): True labels for test data (indexed)
        df (pd.DataFrame): Original dataframe with URLs (must have same index as X_test)
    """
    # Select 8 phishing and 7 legitimate samples
    phishing_mask = y_test == 1
    legitimate_mask = y_test == 0
    
    phishing_indices = y_test[phishing_mask].index[:8].tolist()
    legitimate_indices = y_test[legitimate_mask].index[:7].tolist()
    
    sample_indices = phishing_indices + legitimate_indices
    
    # Get features and predictions
    X_sample = X_test.loc[sample_indices]
    y_sample = y_test.loc[sample_indices]
    y_pred = model.predict(X_sample)
    
    # Get URLs
    urls = df.loc[sample_indices, "URL"].values
    
    # Print header
    print("\n" + "="*60)
    print("SAMPLE PREDICTIONS")
    print("="*60)
    print()
    
    # Print table header
    print(f"{'#':<3} | {'URL (max 55 chars)':<55} | {'Actual':<11} | {'Predicted':<11} | {'Correct?'}")
    print("-"*103)
    
    # Print rows
    correct_count = 0
    misclassified = []
    
    for idx, (i, url, actual, pred) in enumerate(zip(sample_indices, urls, y_sample, y_pred), 1):
        # Truncate URL to 55 chars
        if len(url) > 55:
            url_display = url[:52] + "..."
        else:
            url_display = url
        
        actual_label = LABEL_NAMES[actual]
        pred_label = LABEL_NAMES[pred]
        
        is_correct = actual == pred
        correct_marker = "✓" if is_correct else "✗"
        
        if is_correct:
            correct_count += 1
        else:
            misclassified.append((url, actual_label, pred_label))
        
        print(f"{idx:<3} | {url_display:<55} | {actual_label:<11} | {pred_label:<11} | {correct_marker:>7}")
    
    print("-"*103)
    total_samples = len(sample_indices)
    print(f"Correct: {correct_count}/{total_samples}")
    
    # Print misclassified URLs
    if misclassified:
        print("\nMisclassified URLs:")
        for url, actual, pred in misclassified:
            print(f"  {url}")
            print(f"    Actual: {actual} | Predicted: {pred}")
    
    print("="*60 + "\n")
