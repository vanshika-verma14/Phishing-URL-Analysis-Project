# Paths
DATA_PATH = "data/phishing_site_urls.csv"
MODEL_DIR = "models/"
OUTPUT_DIR = "outputs/"

# Feature thresholds used in rule-based classifier (Phase 3)
THRESHOLD_URL_LENGTH = 75
THRESHOLD_SPECIAL_CHARS = 4
THRESHOLD_DOT_COUNT = 4
THRESHOLD_ENTROPY = 4.0
THRESHOLD_SUBDOMAIN_DEPTH = 3
RULE_SCORE_CUTOFF = 3

# ML config
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Risk score bands
RISK_LOW_MAX = 40
RISK_SUSPICIOUS_MAX = 70

# Label mapping
LABEL_MAP = {"good": 0, "bad": 1}
LABEL_NAMES = {0: "LEGITIMATE", 1: "PHISHING"}
TRANCO_LIST_PATH = "data/top-1m.csv"
SUSPICIOUS_TLDS = {
    '.tk', '.ml', '.ga', '.cf', '.xyz', '.top', '.click',
    '.loan', '.win', '.download', '.racing', '.online',
    '.site', '.club', '.icu', '.monster', '.buzz'
}