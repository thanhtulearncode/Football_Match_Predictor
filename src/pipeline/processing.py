import pandas as pd
import logging
import pandera as pa
from sklearn.preprocessing import OrdinalEncoder
from config.settings import settings
from src.pipeline.validation import RawMatchSchema

logger = logging.getLogger(__name__)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes team names, converts dates, and validates schema."""
    if df.empty:
        logger.warning("Received empty DataFrame for cleaning.")
        return df

    df = df.copy()
    
    name_map = {
        alias: canonical
        for canonical, aliases in settings.TEAM_NAME_MAPPINGS.items()
        for alias in aliases
    }
    
    team_columns = ["team", "opponent", "home_team", "away_team"]
    for col in team_columns:
        if col in df.columns:
            df[col] = df[col].replace(name_map)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        
    if "result" in df.columns:
        df["target"] = df["result"].map({"W": 2, "D": 1, "L": 0})
    
    if "venue" in df.columns:
        df["venue"] = df["venue"].astype(str).str.title()
    
    df = df.dropna(subset=["team", "opponent", "date"])

    if "target" in df.columns:
        df["target"] = df["target"].astype("Int64")
    
    try:
        RawMatchSchema.validate(df, lazy=True)
        logger.info("Data validation passed")
    except pa.errors.SchemaErrors as e:
        logger.warning(f"Data validation issues found (non-critical): {len(e.failure_cases)} cases")
    except Exception as e:
        logger.error(f"Critical validation error: {e}", exc_info=True)
        raise

    return df

def preprocess_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based features (day_code, hour) from date and time columns."""
    df = df.copy()
    
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df["day_code"] = df["date"].dt.dayofweek.astype(int)
    
    if "time" in df.columns:
        # Extract hour from time strings like "15:00", "20:45"
        # Vectorized approach: split by colon and take first part
        df["hour"] = (
            df["time"].astype(str)
            .str.split(":", n=1, expand=True)[0]
            .str.extract(r"(\d+)", expand=False)
            .fillna(15)
            .astype(int)
            .clip(0, 23)
        )
    else:
        df["hour"] = 15
        
    return df

def get_fitted_encoders(df: pd.DataFrame):
    """
    Fits OrdinalEncoders on the provided dataframe.
    Returns a dictionary of fitted encoders.
    """
    encoders = {}
    
    # Encoder for Venue
    venue_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    venue_enc.fit(df[["venue"]])
    encoders["venue"] = venue_enc
    
    # Encoder for Opponent
    opp_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    opp_enc.fit(df[["opponent"]])
    encoders["opponent"] = opp_enc
    
    return encoders

def apply_encoders(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """ Applies the fitted encoders to the dataframe."""
    df = df.copy()
    
    if "venue" in encoders and "venue" in df.columns:
        df["venue_code"] = encoders["venue"].transform(df[["venue"]]).astype(int)
        
    if "opponent" in encoders and "opponent" in df.columns:
        df["opp_code"] = encoders["opponent"].transform(df[["opponent"]]).astype(int)
        
    return df