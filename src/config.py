from pathlib import Path
from typing import List, Final, Dict, Any

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
for directory in [RAW_DIR, PROCESSED_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Files
RAW_DATA_FILE = RAW_DIR / "matches.csv"
TRAIN_DATA_FILE = PROCESSED_DIR / "train.csv"
MODEL_COMPARISON_FILE = MODELS_DIR / "model_comparison.csv"
TEAM_STATS_FILE = PROCESSED_DIR / "team_stats.json"
UPCOMING_MATCHES_FILE = PROCESSED_DIR / "upcoming_matches.csv"

# API Settings
API_HOST: Final[str] = "127.0.0.1"
API_PORT: Final[int] = 8000

# Preprocessing Params
ELO_K_FACTOR: Final[int] = 20
FORM_MATCHES_N: Final[int] = 5
H2H_MATCHES_N: Final[int] = 5
GOAL_STAT_MATCHES: Final[int] = 5

# Model Features
TARGET_COLUMN: Final[str] = "result"
METADATA_COLS: Final[List[str]] = ["date", "home_team", "away_team", "result"]
TEST_SIZE: Final[float] = 0.2
RANDOM_STATE: Final[int] = 42

# Features
FEATURES: Final[List[str]] = [
    'home_elo', 'away_elo', 'elo_diff', 
    'home_form', 'away_form', 'form_diff',
    'home_avg_scored', 'home_avg_conceded',
    'away_avg_scored', 'away_avg_conceded', 
    'attack_strength_diff',
    'h2h_home_wins', 'h2h_draws', 'h2h_away_wins',
    'home_rest_days', 'away_rest_days', 'rest_advantage'
]

# Team name mappings for different data sources
TEAM_NAME_MAPPINGS: Final[Dict[str, List[str]]] = {
    'Manchester United': ['Manchester United', 'Man United', 'Man Utd', 'Manchester Utd'],
    'Manchester City': ['Manchester City', 'Man City'],
    'Newcastle United': ['Newcastle United', 'Newcastle'],
    'Tottenham Hotspur': ['Tottenham Hotspur', 'Tottenham', 'Spurs'],
    'Wolverhampton Wanderers': ['Wolverhampton Wanderers', 'Wolves'],
    'Brighton & Hove Albion': ['Brighton & Hove Albion', 'Brighton'],
    'Nottingham Forest': ['Nottingham Forest', 'Nottm Forest'],
    'Sheffield United': ['Sheffield United', 'Sheffield Utd'],
    'Luton Town': ['Luton Town', 'Luton'],
    'West Ham United': ['West Ham United', 'West Ham'],
    'AFC Bournemouth': ['AFC Bournemouth', 'Bournemouth'],
}

# Football data API configuration
FOOTBALL_DATA_COMPETITIONS: Final[Dict[str, str]] = {
    'PL': 'Premier League',
    'PD': 'La Liga',
    'BL1': 'Bundesliga',
    'SA': 'Serie A',
    'FL1': 'Ligue 1'
}