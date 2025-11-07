"""Configuration file for Football Match Predictor"""
from pathlib import Path
from typing import List, Final

# ===== Directory Paths =====
# Base directory (project root)
BASE_DIR = Path(__file__).resolve().parent.parent
# Data directories for storing raw and processed data
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
# Models directory for saving trained models
MODELS_DIR = BASE_DIR / "models"
# Ensure all required directories exist
for directory in [RAW_DIR, PROCESSED_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
# ===== File Paths =====
# Raw match data file
RAW_DATA_FILE = RAW_DIR / "matches.csv"
# Processed training data file
TRAIN_DATA_FILE = PROCESSED_DIR / "train.csv"
# Model comparison results file
MODEL_COMPARISON_FILE = MODELS_DIR / "model_comparison.csv"
# ===== API Settings =====
# API server host and port
API_HOST: Final[str] = "127.0.0.1"
API_PORT: Final[int] = 8000
# ===== Preprocessing Parameters =====
# Elo rating K-factor (controls how much ratings change per match)
ELO_K_FACTOR: Final[int] = 20
# Number of recent matches to consider for form calculation
FORM_MATCHES_N: Final[int] = 5
# Number of head-to-head matches to consider
H2H_MATCHES_N: Final[int] = 5
# Number of matches to consider for goal statistics
GOAL_STAT_MATCHES: Final[int] = 5
# ===== Model Configuration =====
# Target column name for predictions (0=away win, 1=draw, 2=home win)
TARGET_COLUMN: Final[str] = "result"
# Metadata columns to exclude from feature set
METADATA_COLS: Final[List[str]] = ["date", "home_team", "away_team", "result"]
# Test set size (20% of data)
TEST_SIZE: Final[float] = 0.2
# Random seed for reproducibility
RANDOM_STATE: Final[int] = 42
# ===== Feature List =====
# All features used for model training and prediction
FEATURES: Final[List[str]] = [
    'home_elo', 'away_elo', 'elo_diff',  # Elo ratings
    'home_form', 'away_form', 'form_diff',  # Team form
    'home_avg_scored', 'home_avg_conceded',  # Home team goal stats
    'away_avg_scored', 'away_avg_conceded',  # Away team goal stats
    'attack_strength_diff',  # Attack vs defense strength difference
    'h2h_home_wins', 'h2h_draws', 'h2h_away_wins',  # Head-to-head history
    'home_rest_days', 'away_rest_days', 'rest_advantage'  # Rest days
]