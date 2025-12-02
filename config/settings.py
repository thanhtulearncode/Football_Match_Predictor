import os
from pathlib import Path
from typing import Dict, List, Set
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    # Data Paths
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    MODELS_DIR: Path = BASE_DIR / "models"
    
    # File Names
    RAW_MATCHES_FILE: str = "matches.csv"
    UPCOMING_RAW_FILE: str = "upcoming_matches_raw.csv"
    MATCHES_CLEANED_FILE: str = "matches_cleaned.csv"
    MATCHES_FINAL_FILE: str = "matches_final.csv"
    UPCOMING_FINAL_FILE: str = "upcoming_final.csv"
    MODEL_FILE: str = "best_model.pkl"
    ENCODERS_FILE: str = "encoders.pkl"
    
    # File Paths (computed properties)
    @property
    def RAW_MATCHES_PATH(self) -> Path:
        return self.RAW_DIR / self.RAW_MATCHES_FILE
    
    @property
    def UPCOMING_RAW_PATH(self) -> Path:
        return self.RAW_DIR / self.UPCOMING_RAW_FILE
    
    @property
    def MATCHES_CLEANED_PATH(self) -> Path:
        return self.PROCESSED_DIR / self.MATCHES_CLEANED_FILE
    
    @property
    def MATCHES_FINAL_PATH(self) -> Path:
        return self.PROCESSED_DIR / self.MATCHES_FINAL_FILE
    
    @property
    def UPCOMING_FINAL_PATH(self) -> Path:
        return self.PROCESSED_DIR / self.UPCOMING_FINAL_FILE
    
    @property
    def MODEL_PATH(self) -> Path:
        return self.MODELS_DIR / self.MODEL_FILE
    
    @property
    def ENCODERS_PATH(self) -> Path:
        return self.MODELS_DIR / self.ENCODERS_FILE
    
    # External APIs
    FOOTBALL_DATA_API_KEY: str = ""
    FBREF_BASE_URL: str = "https://fbref.com"
    FOOTBALL_API_BASE_URL: str = "http://api.football-data.org/v4"
    
    # Application Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    APP_TITLE: str = "Football Match Predictor"
    VERSION: str = "2.0.0"
    
    # Model Configuration
    SPLIT_DATE: str = "2025-01-01"
    MIN_ACCURACY_THRESHOLD: float = 0.50
    
    # Feature Engineering Configuration
    # Elo Rating Configuration
    ELO_K_FACTOR: int = 20
    ELO_INITIAL_RATING: float = 1500.0
    ELO_TARGET_TO_SCORE: Dict[int, float] = {2: 1.0, 1: 0.5, 0: 0.0}
    
    # Rolling Statistics Configuration
    ROLLING_WINDOW: int = 3
    ROLLING_STATS_COLS: List[str] = [
        "gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"
    ]
    
    # Model Predictors
    PREDICTORS: List[str] = [
        "venue_code", "opp_code", "hour", "day_code",
        "elo", "elo_opp",
        "gf_rolling", "ga_rolling", "sh_rolling", "sot_rolling",
        "dist_rolling", "fk_rolling", "pk_rolling", "pkatt_rolling",
        "gf_rolling_opp", "ga_rolling_opp", "sh_rolling_opp", "sot_rolling_opp",
        "dist_rolling_opp", "fk_rolling_opp", "pk_rolling_opp", "pkatt_rolling_opp"
    ]
    
    # Data Processing Configuration
    HOME_VENUES: Set[str] = {"Home", "home", "H"}
    
    # Team Name Mappings (canonical name -> list of aliases)
    TEAM_NAME_MAPPINGS: Dict[str, List[str]] = {
        "Manchester United": ["Man United", "Manchester Utd"],
        "Newcastle United": ["Newcastle Utd", "Newcastle"],
        "Tottenham Hotspur": ["Spurs", "Tottenham"],
        "Wolverhampton Wanderers": ["Wolves", "Wolverhampton"],
        "Brighton and Hove Albion": ["Brighton"],
        "West Ham United": ["West Ham"],
        "Nottingham Forest": ["Nott'm Forest"],
        "Sheffield United": ["Sheffield Utd"],
        "Luton Town": ["Luton"],
        "Leeds United": ["Leeds"],
        "Leicester City": ["Leicester"],
        "Norwich City": ["Norwich"],
        "Watford": [],
        "Burnley": [],
        "Brentford": [],
        "Aston Villa": [],
        "Crystal Palace": [],
        "Chelsea": [],
        "Arsenal": [],
        "Liverpool": [],
        "Manchester City": ["Man City"],
        "Everton": [],
        "Southampton": [],
        "Bournemouth": [],
        "Fulham": []
    }
    
    # Pipeline Configuration
    PIPELINE_STEPS: int = 5
    SCRAPING_START_YEAR: int = 2025
    SCRAPING_END_YEAR: int = 2023
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"  # Ignore extra variables in .env
    )
    
    def ensure_directories(self):
        """Create necessary directories if they don't exist."""
        for path in [self.RAW_DIR, self.PROCESSED_DIR, self.MODELS_DIR]:
            path.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
settings.ensure_directories()