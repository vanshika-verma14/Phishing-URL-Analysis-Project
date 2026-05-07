# Phishing URL Detection System

A machine learning system that classifies URLs as phishing or legitimate using rule-based logic, multiple ML models, explainable predictions, and risk scoring. Built with Python and Streamlit.

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

```
phishing-detector/
├── data/                        # Place dataset files here (not included)
├── models/                      # Auto-created after training
├── outputs/                     # Auto-created, stores plots
├── src/
│   ├── __init__.py
│   ├── eda.py                   # Data loading utility
│   ├── features.py              # Feature extraction from URLs
│   ├── rule_based.py            # Threshold-based classifier
│   ├── ml_models.py             # ML training, evaluation, save/load
│   └── explainer.py             # Explainability and risk scoring
├── config.py                    # All thresholds and constants
├── train.py                     # Run once to train all models
├── main.py                      # CLI live URL analyser
├── app.py                       # Streamlit web UI
├── eda_analysis.ipynb           # Jupyter EDA notebook
└── requirements.txt
```

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/vanshika-verma14/Phishing-URL-Analysis-Project.git
cd Phishing-URL-Analysis-Project
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download required data files

You need two files placed inside the `data/` folder. Create the folder if it does not exist:

```bash
mkdir data
```

**File 1 — Phishing Dataset:**
- Go to: https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls
- You need a free Kaggle account
- Download and extract the zip
- Place `phishing_site_urls.csv` inside `data/`

**File 2 — Tranco Domain Whitelist:**
- Go to: https://tranco-list.eu/top-1m.csv.zip
- Download and extract the zip
- Place `top-1m.csv` inside `data/`

Your `data/` folder should look like this:

```
data/
├── phishing_site_urls.csv
└── top-1m.csv
```

### 5. Train the model

```bash
python train.py
```

This takes 2–3 minutes. It will:
- Load and explore the dataset
- Extract features from all 549,000 URLs
- Run rule-based classifier and show results
- Train and compare 4 ML models
- Tune classification threshold using precision-recall curve
- Save best model to `models/`
- Save all confusion matrix plots to `outputs/`

### 6. Run the web app

```bash
streamlit run app.py
```

Opens automatically at `http://localhost:8501`

### 7. Run EDA notebook (optional)

```bash
jupyter notebook
```

Open `eda_analysis.ipynb` and click Run All Cells to see full visual analysis.

### 8. Run CLI analyser (optional)

```bash
python main.py
```

Or analyse any URL directly in Python:

```python
from main import analyse
analyse("http://suspicious-login.xyz/verify?id=1234")
```

---

## Model Results

| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.7755   | 0.7819    | 0.7755 | 0.7338   |
| Decision Tree       | 0.8296   | 0.8242    | 0.8296 | 0.8248   |
| Random Forest       | 0.8418   | 0.8371    | 0.8418 | 0.8370   |
| XGBoost             | 0.8293   | 0.8290    | 0.8293 | 0.8149   |

Best model: **Random Forest** with 84.18% accuracy.  
Rule-based baseline accuracy: 67.43%  
Tuned classification threshold: 0.4316 (optimised via precision-recall curve)

---

## Sample Output

```
==========================================
PHISHING DETECTION REPORT
==========================================
URL      : http://paypa1-secure-login.xyz/account/confirm
------------------------------------------
Rule-Based Verdict : PHISHING
ML Verdict         : PHISHING
Risk Score         : 87.3/100 — HIGH RISK
------------------------------------------
Why flagged:
  - No HTTPS — connection is not secure
  - Suspicious TLD detected (.xyz)
  - High entropy (4.81) — randomly generated domain
  - High special character count (5)
------------------------------------------
Feature Breakdown:
  URL Length       : 52
  HTTPS Present    : No
  Special Chars    : 5
  Dot Count        : 2
  Has IP Address   : No
  Digit Count      : 1
  Subdomain Depth  : 1
  Shannon Entropy  : 4.81
==========================================
```

---

## Tech Stack

Python, scikit-learn, XGBoost, pandas, numpy, seaborn, matplotlib, joblib, streamlit, jupyter

---

## Dataset Sources

- Phishing URLs: https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls
- Domain Whitelist: https://tranco-list.eu

---

## Notes

- `models/` and `data/` are not included in the repository due to file size limits
- Run `train.py` after cloning to generate all required model and output files
- Accuracy is capped at ~84% by design — URL-only features cannot catch all phishing sites without visiting actual page content
- The Tranco whitelist ensures popular legitimate sites like google.com, claude.ai, github.com are never falsely flagged
