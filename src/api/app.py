import sys
from pathlib import Path
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from typing import Optional, Tuple

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from config.settings import settings
from src.api.schemas import MatchRequest
from src.pipeline.processing import preprocess_time_features, apply_encoders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("API")

app = FastAPI(title=settings.APP_TITLE, version=settings.VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for loaded artifacts
_model: Optional[object] = None
_encoders: Optional[dict] = None


def _load_artifact(path: Path, artifact_type: str) -> Optional[object]:
    """Load a single artifact (model or encoders) from disk."""
    if not path.exists():
        logger.warning(f"{artifact_type} file not found at {path}")
        return None
    try:
        artifact = joblib.load(path)
        logger.info(f"{artifact_type} loaded from {path}")
        return artifact
    except Exception as e:
        logger.error(f"Failed to load {artifact_type}: {e}")
        return None


@app.on_event("startup")
def load_artifacts():
    """Load model and encoders on application startup."""
    global _model, _encoders
    _model = _load_artifact(settings.MODEL_PATH, "Model")
    _encoders = _load_artifact(settings.ENCODERS_PATH, "Encoders")


def _validate_artifacts_loaded() -> None:
    """Raise HTTPException if artifacts are not loaded."""
    if not _model or not _encoders:
        raise HTTPException(
            status_code=503,
            detail="Model or encoders not loaded. Please check server logs."
        )


def get_class_indices(classes) -> Tuple[int, int, int]:
    """Extract class indices for Win/Draw/Loss or return default numeric mapping."""
    class_list = list(classes)
    try:
        return (
            class_list.index("W"),
            class_list.index("D"),
            class_list.index("L")
        )
    except ValueError:
        return (2, 1, 0)


def format_prediction(pred: float, probs: list, classes) -> dict:
    """Format prediction result with probabilities in a standardized structure."""
    win_idx, draw_idx, loss_idx = get_class_indices(classes)
    
    prediction_map = {2.0: "Home Win", 1.0: "Draw", 0.0: "Away Win"}
    prediction_str = prediction_map.get(float(pred), "Draw")
    
    return {
        "prediction": prediction_str,
        "confidence": float(max(probs)),
        "probabilities": {
            "home_win": float(probs[win_idx]),
            "draw": float(probs[draw_idx]),
            "away_win": float(probs[loss_idx])
        }
    }


def load_upcoming_data() -> pd.DataFrame:
    """Load and return upcoming matches data."""
    if not settings.UPCOMING_FINAL_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="No upcoming matches data found. Please run the data pipeline."
        )
    return pd.read_csv(settings.UPCOMING_FINAL_PATH)


def filter_home_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Filter dataframe to only include home matches."""
    if "venue" not in df.columns:
        return df
    
    return df[df["venue"].isin(settings.HOME_VENUES)]


def prepare_prediction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare dataframe for prediction by applying preprocessing steps."""
    df = preprocess_time_features(df)
    df = apply_encoders(df, _encoders)
    return df


def build_prediction_results(df: pd.DataFrame, preds: list, probs: list) -> list:
    """Build list of prediction result dictionaries from dataframe and model outputs."""
    results = []
    for (_, row), pred, prob in zip(df.iterrows(), preds, probs):
        result = format_prediction(pred, prob, _model.classes_)
        results.append({
            "date": str(row["date"]),
            "time": row.get("time", "00:00"),
            "home_team": row["team"],
            "away_team": row["opponent"],
            **result
        })
    return results


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": _model is not None,
        "encoders_loaded": _encoders is not None
    }


@app.get("/teams")
def get_teams():
    """Return list of unique teams from upcoming matches data."""
    try:
        if not settings.UPCOMING_FINAL_PATH.exists():
            return {"teams": []}
        
        df = load_upcoming_data()
        teams = sorted(set(df["team"].unique()) | set(df["opponent"].unique()))
        return {"teams": teams}
    except Exception as e:
        logger.error(f"Error fetching teams: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch teams")


@app.get("/predict/upcoming")
def predict_upcoming():
    """Return predictions for all upcoming matches."""
    _validate_artifacts_loaded()
    
    try:
        df = load_upcoming_data()
        df = filter_home_matches(df)
        
        if df.empty:
            return {"predictions": [], "message": "No home fixtures found."}

        df = prepare_prediction_features(df)
        X = df[settings.PREDICTORS].fillna(0)
        
        preds = _model.predict(X)
        probs = _model.predict_proba(X)
        
        results = build_prediction_results(df, preds, probs)
        return {"predictions": results}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate predictions")


def merge_opponent_stats(home_row: pd.DataFrame, away_row: pd.DataFrame) -> pd.DataFrame:
    """Merge opponent rolling statistics into home row dataframe."""
    match_features = home_row.copy()
    
    for col in settings.ROLLING_STATS_COLS:
        col_name = f"{col}_rolling"
        if col_name in away_row.columns:
            match_features[f"{col_name}_opp"] = away_row.iloc[0][col_name]
    
    if "elo" in away_row.columns:
        match_features["elo_opp"] = away_row.iloc[0]["elo"]
    
    return match_features


@app.post("/predict/teams")
def predict_custom_match(request: MatchRequest):
    """Predict match outcome between two specific teams using their latest stats."""
    _validate_artifacts_loaded()

    try:
        df = load_upcoming_data()
        
        home_row = df[df["team"] == request.home_team].sort_values("date").iloc[-1:]
        away_row = df[df["team"] == request.away_team].sort_values("date").iloc[-1:]
        
        if home_row.empty or away_row.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Team data not found for {request.home_team} or {request.away_team}"
            )
        
        match_features = merge_opponent_stats(home_row, away_row)
        
        match_features["venue"] = "Home"
        match_features["opponent"] = request.away_team
        match_features["hour"] = request.hour
        match_features["day_code"] = request.day_code
        
        match_features = apply_encoders(match_features, _encoders)
        X = match_features[settings.PREDICTORS].fillna(0)
        
        pred = _model.predict(X)[0]
        probs = _model.predict_proba(X)[0]
        result = format_prediction(pred, probs, _model.classes_)
        
        return {
            "home_team": request.home_team,
            "away_team": request.away_team,
            **result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Custom prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate custom prediction")