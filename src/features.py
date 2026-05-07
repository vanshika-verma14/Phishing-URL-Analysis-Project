"""
Feature extraction module for phishing detection.

This module is responsible for engineering features from URLs for both training
and live prediction. Features are designed to capture patterns that distinguish
phishing URLs from legitimate ones.
"""

import re
import math
import time
from urllib.parse import urlparse
import pandas as pd
import numpy as np

from config import TRANCO_LIST_PATH

# Cache for Tranco domains - loaded only once
TRANCO_DOMAINS = None


def load_tranco_domains() -> set:
    global TRANCO_DOMAINS
    if TRANCO_DOMAINS is not None:
        return TRANCO_DOMAINS
    try:
        df = pd.read_csv(TRANCO_LIST_PATH, header=None, names=["rank", "domain"])
        TRANCO_DOMAINS = set(df["domain"].str.lower().str.strip())
        print(f"Tranco whitelist loaded: {len(TRANCO_DOMAINS)} domains")
        return TRANCO_DOMAINS
    except FileNotFoundError:
        print("WARNING: Tranco list not found at", TRANCO_LIST_PATH)
        TRANCO_DOMAINS = set()
        return TRANCO_DOMAINS
    except Exception as e:
        print(f"WARNING: Could not load Tranco list: {e}")
        TRANCO_DOMAINS = set()
        return TRANCO_DOMAINS


def is_whitelisted(url: str) -> bool:
    try:
        from urllib.parse import urlparse
        netloc = urlparse(url).netloc.lower()
        netloc = netloc.replace("www.", "")
        # remove port if present e.g. example.com:8080
        netloc = netloc.split(":")[0]
        domains = load_tranco_domains()
        if netloc in domains:
            return True
        # check parent domain for subdomains e.g. mail.google.com
        parts = netloc.split(".")
        if len(parts) > 2:
            parent = ".".join(parts[-2:])
            if parent in domains:
                return True
        return False
    except Exception:
        return False

# Define features extracted from URLs
FEATURE_COLUMNS = [
    "url_length", "has_https", "special_char_count",
    "dot_count", "digit_count", "has_ip_address",
    "subdomain_depth", "shannon_entropy"
]


def extract_features(url: str) -> dict:
    """
    Extract all 8 features from a single URL string.
    
    Features extracted:
    - url_length: Length of the URL. Phishing URLs are often artificially long
      to obfuscate the actual domain.
    - has_https: Binary flag. Legitimate sites typically use HTTPS; phishing 
      sites often use HTTP.
    - special_char_count: Count of suspicious special characters (@, -, _, ?, =, 
      &, %, #, ~). Phishing URLs often contain many special chars to confuse users.
    - dot_count: Number of dots in URL. Phishing URLs often have excessive 
      subdomains (more dots).
    - digit_count: Count of digits. Phishing URLs often use IPs or numeric 
      obfuscation.
    - has_ip_address: Binary flag. Direct IP addresses are rarely used in 
      legitimate URLs but common in phishing.
    - subdomain_depth: Count of dots in the netloc (domain) part only. Excessive 
      subdomains indicate phishing attempts.
    - shannon_entropy: Entropy of character distribution. Phishing URLs often 
      have high entropy due to random characters.
    
    Args:
        url (str): The URL string to extract features from.
        
    Returns:
        dict: Dictionary with all 8 features. Returns zeros for edge cases 
              (empty/malformed URLs) without raising exceptions.
    """
    features = {
        "url_length": 0,
        "has_https": 0,
        "special_char_count": 0,
        "dot_count": 0,
        "digit_count": 0,
        "has_ip_address": 0,
        "subdomain_depth": 0,
        "shannon_entropy": 0.0
    }
    
    # Handle edge case: empty string
    if not url or not isinstance(url, str):
        return features
    
    try:
        # Basic features
        features["url_length"] = len(url)
        features["has_https"] = 1 if url.startswith("https") else 0
        
        # Special character count
        special_chars = "@-_?=&%#~"
        features["special_char_count"] = sum(url.count(char) for char in special_chars)
        
        # Dot count
        features["dot_count"] = url.count(".")
        
        # Digit count
        features["digit_count"] = sum(1 for c in url if c.isdigit())
        
        # IP address detection
        ip_pattern = r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"
        features["has_ip_address"] = 1 if re.search(ip_pattern, url) else 0
        
        # Subdomain depth (dots in netloc only)
        try:
            parsed = urlparse(url)
            netloc = parsed.netloc if parsed.netloc else url
            features["subdomain_depth"] = netloc.count(".")
        except Exception:
            features["subdomain_depth"] = 0
        
        # Shannon entropy
        try:
            char_counts = {}
            for char in url:
                char_counts[char] = char_counts.get(char, 0) + 1
            
            entropy = 0.0
            url_len = len(url)
            for count in char_counts.values():
                p = count / url_len
                entropy -= p * math.log2(p)
            features["shannon_entropy"] = entropy
        except Exception:
            features["shannon_entropy"] = 0.0
    
    except Exception:
        # Return zeros for any unexpected error
        pass
    
    return features


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build feature matrix from a dataframe of URLs.
    
    Applies feature extraction to every URL in the dataframe and generates
    a statistical summary grouped by label to show patterns.
    
    Args:
        df (pd.DataFrame): Dataframe with "URL" and "label" columns.
        
    Returns:
        pd.DataFrame: Dataframe with FEATURE_COLUMNS as columns. One row per URL.
    """
    start_time = time.time()
    
    # Extract features for all URLs
    features_list = [extract_features(url) for url in df["URL"]]
    features_df = pd.DataFrame(features_list)
    
    elapsed_time = time.time() - start_time
    print(f"\nFeature extraction completed in {elapsed_time:.2f} seconds.")
    
    # Add labels for grouping
    features_df["label"] = df["label"].values
    
    # Print feature statistics grouped by label
    print("\n--- Feature Statistics by Class ---")
    stats = features_df.groupby("label")[FEATURE_COLUMNS].mean()
    
    # Rename rows for readability
    stats.index = ["LEGITIMATE", "PHISHING"]
    
    print(stats.round(3))
    
    # Return feature matrix without label column
    return features_df[FEATURE_COLUMNS]


def get_feature_vector(url: str) -> pd.DataFrame:
    """
    Extract features for a single URL in prediction-ready format.
    
    This function is designed for live prediction. It extracts features
    from a single URL and returns them as a single-row DataFrame that
    matches the format expected by the model.
    
    Args:
        url (str): The URL string to extract features from.
        
    Returns:
        pd.DataFrame: Single-row DataFrame with FEATURE_COLUMNS as columns.
    """
    features = extract_features(url)
    # Create DataFrame with features in correct column order
    feature_vector = pd.DataFrame([{col: features[col] for col in FEATURE_COLUMNS}])
    return feature_vector
