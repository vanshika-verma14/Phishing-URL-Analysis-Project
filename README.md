# Phishing URL Detection System

A machine learning system that classifies URLs as phishing or legitimate using 
rule-based logic, multiple ML models, explainable predictions, and risk scoring.
Built with Python and Streamlit.

---

## Features

- Rule-based classifier using URL feature thresholds
- ML classification with Logistic Regression, Decision Tree, Random Forest, XGBoost
- Feature extraction from raw URLs (length, HTTPS, entropy, special chars, IP detection)
- Tranco Top 1M domain whitelist for zero false positives on popular sites
- Suspicious TLD detection (.tk, .ml, .ga, .xyz etc.)
- Explainable predictions — shows exactly why a URL was flagged
- Risk scoring (0–100) with LOW / SUSPICIOUS / HIGH RISK levels
- Clean Streamlit web UI
- Jupyter notebook for full exploratory data analysis

---

## Project Structure
