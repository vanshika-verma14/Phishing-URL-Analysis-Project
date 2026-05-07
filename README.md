# Phishing Detection System

A machine learning system that classifies URLs as phishing or legitimate, with explainable predictions and risk scoring.

## Features

- **Dual classification**: Rule-based and ML approaches (Logistic Regression, Decision Tree, Random Forest)
- **Feature extraction**: 8 engineered features from raw URLs (length, HTTPS, entropy, special characters, IP detection, subdomain depth, and more)
- **Explainable predictions**: Shows exactly which rules and patterns triggered the phishing detection
- **Risk scoring**: 0-100 risk score on top of binary classification for nuanced threat assessment
- **Clean modular codebase**: Each component is independently usable and testable

## Project Structure

```
phishing-detector/
├── data/                        # Place dataset here
├── models/                      # Saved model files (auto-created)
├── outputs/                     # Plots and charts (auto-created)
├── src/
│   ├── __init__.py              # Package initialization
│   ├── eda.py                   # Data loading and exploration
│   ├── features.py              # Feature extraction from URLs
│   ├── rule_based.py            # Threshold-based classifier
│   ├── ml_models.py             # ML training, evaluation, save/load
│   └── explainer.py             # Explainability and risk scoring
├── config.py                    # All thresholds and constants
├── train.py                     # Run once to train the system
├── main.py                      # Run to analyse URLs live
└── requirements.txt             # Python dependencies
```

## Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/yourusername/phishing-detector.git
   cd phishing-detector
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download dataset**
   Download from [Kaggle](https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls) and place `phishing_site_urls.csv` in the `data/` folder

4. **Train the model**
   ```bash
   python train.py
   ```

5. **Start analyzing URLs**
   ```bash
   python main.py
   ```

## Usage

### Programmatic use
```python
from main import analyse

# Analyze a URL
prediction = analyse("http://suspicious-login.xyz/verify?id=1234")
# Returns: 1 for phishing, 0 for legitimate
```

### Interactive use
Simply run:
```bash
python main.py
```

This will demonstrate the detector on sample URLs and show you how to use it.

## Output Example

```
==========================================
PHISHING DETECTION REPORT
==========================================
URL      : http://suspicious-login.xyz/verify?id=1234
------------------------------------------
Rule-Based Verdict : PHISHING
ML Verdict         : PHISHING
Risk Score         : 91.2/100 — HIGH RISK
------------------------------------------
Why flagged:
  - No HTTPS — connection is not secure
  - High entropy (4.81) — suggests randomly generated domain
  - High special character count (5)
------------------------------------------
Feature Breakdown:
  URL Length       : 43
  HTTPS Present    : No
  Special Chars    : 5
  Dot Count        : 2
  Has IP Address   : No
  Digit Count      : 4
  Subdomain Depth  : 1
  Shannon Entropy  : 4.81
==========================================
```

## Results

Expected performance (results vary slightly between runs):

- **Rule-Based Classifier**: ~72-75% accuracy
- **Logistic Regression**: ~94-95% accuracy
- **Decision Tree**: ~93-94% accuracy
- **Random Forest**: ~96-97% accuracy (selected for production)

All confusion matrices and evaluation plots are saved to `outputs/` during training.

## Tech Stack

- **Python 3.8+**
- **scikit-learn** — ML models and metrics
- **pandas** — Data manipulation
- **numpy** — Numerical operations
- **seaborn** — Data visualization
- **matplotlib** — Plotting
- **joblib** — Model serialization
