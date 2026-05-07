"""
Rule-based classifier module for phishing detection.

This module implements a threshold-based rule classifier that does not require
any trained ML model. It uses feature thresholds from config to make predictions
and can generate detailed explanations of decisions.
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from config import (
    THRESHOLD_URL_LENGTH, THRESHOLD_SPECIAL_CHARS, THRESHOLD_DOT_COUNT,
    THRESHOLD_ENTROPY, THRESHOLD_SUBDOMAIN_DEPTH, RULE_SCORE_CUTOFF, SUSPICIOUS_TLDS
)
from src.features import extract_features


def score_url(feature_dict: dict, url: str = "") -> tuple[int, int, list]:
    """
    Score a URL based on rule-based thresholds.
    
    Implements a scoring system where different features contribute points if they
    exceed predefined thresholds. Features are weighted based on their indicator
    strength for phishing URLs.
    
    Scoring rules:
    - URL length exceeds threshold: +1 (long URLs can obfuscate the actual domain)
    - No HTTPS: +1 (legitimate sites use HTTPS)
    - Excessive special characters: +1 (phishing uses special chars to confuse)
    - IP address present: +2 (strong indicator of phishing)
    - Excess dots: +1 (multiple subdomains indicate phishing)
    - High entropy: +1 (random characters suggest phishing)
    - Deep subdomain structure: +1 (excessive nesting indicates phishing)
    - Suspicious TLD: +2 (known phishing TLDs)
    
    Args:
        feature_dict (dict): Dictionary of features from extract_features().
        url (str): The URL string to check for suspicious TLDs (optional).
        
    Returns:
        tuple: (prediction, score, triggered_rules)
            - prediction: 1 (phishing) if score >= RULE_SCORE_CUTOFF or (score >= 2 AND has_ip_address), else 0
            - score: Total score based on triggered rules
            - triggered_rules: List of human-readable descriptions of rules that fired
    """
    score = 0
    triggered_rules = []
    
    # Rule 1: URL length
    if feature_dict["url_length"] > THRESHOLD_URL_LENGTH:
        score += 1
        triggered_rules.append(f"URL length exceeds threshold ({feature_dict['url_length']})")
    
    # Rule 2: HTTPS check
    if feature_dict["has_https"] == 0:
        score += 1
        triggered_rules.append("No HTTPS detected")
    
    # Rule 3: Special character count
    if feature_dict["special_char_count"] > THRESHOLD_SPECIAL_CHARS:
        score += 1
        triggered_rules.append(f"Excessive special characters ({feature_dict['special_char_count']})")
    
    # Rule 4: IP address detection (high weight)
    if feature_dict["has_ip_address"] == 1:
        score += 2
        triggered_rules.append("Direct IP address detected")
    
    # Rule 5: Dot count
    if feature_dict["dot_count"] > THRESHOLD_DOT_COUNT:
        score += 1
        triggered_rules.append(f"Excessive dots in URL ({feature_dict['dot_count']})")
    
    # Rule 6: Shannon entropy
    if feature_dict["shannon_entropy"] > THRESHOLD_ENTROPY:
        score += 1
        triggered_rules.append(f"High entropy detected ({feature_dict['shannon_entropy']:.2f})")
    
    # Rule 7: Subdomain depth
    if feature_dict["subdomain_depth"] > THRESHOLD_SUBDOMAIN_DEPTH:
        score += 1
        triggered_rules.append(f"Deep subdomain structure ({feature_dict['subdomain_depth']} levels)")
    
    # Rule 8: Suspicious TLD
    if url:
        from urllib.parse import urlparse
        parsed = urlparse(url if url.startswith('http') else 'http://' + url)
        netloc = parsed.netloc.lower()
        for tld in SUSPICIOUS_TLDS:
            if netloc.endswith(tld):
                score += 2
                triggered_rules.append(f"Suspicious TLD detected ({tld})")
                break
    
    # Determine prediction
    # Return phishing if score >= RULE_SCORE_CUTOFF, or if score >= 2 AND has_ip_address
    if score >= RULE_SCORE_CUTOFF or (score >= 2 and feature_dict.get("has_ip_address") == 1):
        prediction = 1
    else:
        prediction = 0
    
    return prediction, score, triggered_rules


def predict(feature_row, url: str = "") -> int:
    """
    Make a single prediction with sklearn-compatible interface.
    
    Accepts either a dictionary or a single-row DataFrame and returns
    the predicted label.
    
    Args:
        feature_row: Either a dict (from extract_features) or a single-row pd.DataFrame
        url (str): The URL string for additional TLD checks (optional).
        
    Returns:
        int: Predicted label (0 for legitimate, 1 for phishing)
    """
    if isinstance(feature_row, pd.DataFrame):
        feature_dict = feature_row.iloc[0].to_dict()
    else:
        feature_dict = feature_row
    
    prediction, _, _ = score_url(feature_dict, url)
    return prediction


def predict_batch(feature_df: pd.DataFrame) -> np.ndarray:
    """
    Make predictions for a batch of URLs.
    
    Applies the rule-based classifier to every row in the feature dataframe.
    
    Args:
        feature_df (pd.DataFrame): DataFrame with feature columns matching FEATURE_COLUMNS
        
    Returns:
        np.ndarray: Array of predictions (0 or 1 for each row)
    """
    predictions = []
    for _, row in feature_df.iterrows():
        pred = predict(row.to_dict())
        predictions.append(pred)
    return np.array(predictions)


def evaluate(feature_df: pd.DataFrame, labels: pd.Series, output_dir: str) -> np.ndarray:
    """
    Evaluate the rule-based classifier and generate comprehensive metrics.
    
    Generates accuracy, classification report, and confusion matrix visualization.
    Metrics are printed to console and confusion matrix is saved as PNG.
    
    Args:
        feature_df (pd.DataFrame): DataFrame with feature columns matching FEATURE_COLUMNS
        labels (pd.Series): True labels (0 for legitimate, 1 for phishing)
        output_dir (str): Directory to save output plots
        
    Returns:
        np.ndarray: Array of predictions made by the classifier
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get predictions
    predictions = predict_batch(feature_df)
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    class_report = classification_report(labels, predictions, target_names=["Legitimate", "Phishing"])
    conf_matrix = confusion_matrix(labels, predictions)
    
    # Print results
    print("\n" + "="*60)
    print("RULE-BASED CLASSIFIER RESULTS")
    print("="*60)
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\n" + class_report)
    
    # Save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Legitimate", "Phishing"],
                yticklabels=["Legitimate", "Phishing"])
    plt.title("Rule-Based Classifier", fontsize=14, fontweight="bold")
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, "rule_based_confusion_matrix.png")
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close()
    
    print(f"Confusion matrix saved to {output_path}")
    
    return predictions


def get_triggered_rules(url: str) -> list:
    """
    Get the list of triggered rules for a given URL.
    
    Extracts features from the URL and identifies which rules fire,
    providing an explanation of why the classifier made its decision.
    
    Args:
        url (str): The URL to analyze
        
    Returns:
        list: List of strings describing which rules were triggered
    """
    features = extract_features(url)
    _, _, triggered_rules = score_url(features)
    return triggered_rules
