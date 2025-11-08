"""FastAPI app for football match predictions"""
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator
import joblib
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from contextlib import asynccontextmanager
import os
import json
from datetime import datetime, timedelta
import asyncio

# Import config
from src.config import MODELS_DIR, FEATURES, TEAM_STATS_FILE, UPCOMING_MATCHES_FILE, PROCESSED_DIR, TEAM_NAME_MAPPINGS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MatchInput(BaseModel):
    """Manual feature input validation"""
    home_elo: float = Field(..., ge=1000, le=2500)
    away_elo: float = Field(..., ge=1000, le=2500)
    home_form: float = Field(..., ge=0, le=15)
    away_form: float = Field(..., ge=0, le=15)
    home_avg_scored: float = Field(..., ge=0)
    home_avg_conceded: float = Field(..., ge=0)
    away_avg_scored: float = Field(..., ge=0)
    away_avg_conceded: float = Field(..., ge=0)
    h2h_home_wins: int = Field(..., ge=0)
    h2h_draws: int = Field(..., ge=0)
    h2h_away_wins: int = Field(..., ge=0)
    home_rest_days: int = Field(..., ge=0, le=30)
    away_rest_days: int = Field(..., ge=0, le=30)

    @model_validator(mode='after')
    def validate_h2h_totals(self):
        """Validate that H2H stats are reasonable"""
        total = self.h2h_home_wins + self.h2h_draws + self.h2h_away_wins
        if total > 20:
            raise ValueError('Total H2H matches too high')
        return self

class TeamPredictionRequest(BaseModel):
    """Request for team-based prediction"""
    home_team: str
    away_team: str
    home_rest_days: int = Field(7, ge=0, le=30)
    away_rest_days: int = Field(7, ge=0, le=30)

class UpcomingMatch(BaseModel):
    """Upcoming match structure"""
    date: str
    home_team: str
    away_team: str
    competition: Optional[str] = None

class PredictionResponse(BaseModel):
    """Prediction response"""
    prediction: str
    prediction_code: int
    probabilities: Dict[str, float]
    confidence: float
    model_used: str
    features_used: Dict[str, float]

class TeamPredictionResponse(BaseModel):
    """Team-based prediction response"""
    home_team: str
    away_team: str
    prediction: str
    prediction_code: int
    probabilities: Dict[str, float]
    confidence: float
    model_used: str
    features_used: Dict[str, float]

class UpcomingPredictionResponse(BaseModel):
    """Upcoming match prediction response"""
    date: str
    home_team: str
    away_team: str
    competition: Optional[str]
    prediction: str
    prediction_code: int
    probabilities: Dict[str, float]
    confidence: float

class ModelManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = None
            cls._instance.model_name = None
            cls._instance.feature_names = None
            cls._instance.team_stats = None
            cls._instance.historical_data = None
            cls._instance.upcoming_matches = None
        return cls._instance
    
    def load_best_model(self) -> bool:
        """Load the best available model with team statistics"""
        try:
            # Try to load from comparison file first
            comparison_path = MODELS_DIR / "model_comparison.csv"
            if comparison_path.exists():
                comparison = pd.read_csv(comparison_path, index_col=0)
                best_model_name = comparison['test_accuracy'].idxmax()
                model_path = MODELS_DIR / f"{best_model_name}_model.pkl"
                logger.info(f"Best model from comparison: {best_model_name}")
            else:
                # Fallback to first available model
                model_candidates = ['XGBoost', 'LightGBM', 'RandomForest', 'GradientBoosting']
                for candidate in model_candidates:
                    candidate_path = MODELS_DIR / f"{candidate}_model.pkl"
                    if candidate_path.exists():
                        best_model_name = candidate
                        model_path = candidate_path
                        logger.info(f"Using available model: {best_model_name}")
                        break
                else:
                    logger.error("No model files found!")
                    return False

            # Load model
            if model_path.exists():
                self.model = joblib.load(model_path)
                self.model_name = best_model_name
                
                # Get feature names
                if hasattr(self.model, 'feature_names_in_'):
                    self.feature_names = self.model.feature_names_in_.tolist()
                else:
                    self.feature_names = FEATURES
                
                logger.info(f"Successfully loaded: {self.model_name} with {len(self.feature_names)} features")
                
                # Load team statistics and data
                self.load_team_stats()
                self.load_historical_data()
                self.load_upcoming_matches()
                
                return True
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
        
        return False

    def load_team_stats(self):
        """Load team statistics"""
        try:
            if TEAM_STATS_FILE.exists():
                with open(TEAM_STATS_FILE, 'r') as f:
                    self.team_stats = json.load(f)
                logger.info(f"Loaded statistics for {len(self.team_stats)} teams")
            else:
                logger.warning("Team statistics file not found")
                self.team_stats = {}
        except Exception as e:
            logger.error(f"Error loading team stats: {e}")
            self.team_stats = {}

    def load_historical_data(self):
        """Load historical data for head-to-head calculations"""
        try:
            train_path = PROCESSED_DIR / "train.csv"
            if train_path.exists():
                self.historical_data = pd.read_csv(train_path, parse_dates=['date'])
                logger.info(f"Loaded historical data: {len(self.historical_data)} matches")
            else:
                logger.warning("Historical data file not found")
                self.historical_data = pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading historical data: {e}")
            self.historical_data = pd.DataFrame()

    def load_upcoming_matches(self):
        """Load upcoming matches"""
        try:
            if UPCOMING_MATCHES_FILE.exists():
                self.upcoming_matches = pd.read_csv(UPCOMING_MATCHES_FILE)
                logger.info(f"Loaded {len(self.upcoming_matches)} upcoming matches")
            else:
                logger.warning("Upcoming matches file not found")
                self.upcoming_matches = pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading upcoming matches: {e}")
            self.upcoming_matches = pd.DataFrame()

    def normalize_team_name(self, team_name: str) -> str:
        """Normalize team name using mappings"""
        for canonical_name, variants in TEAM_NAME_MAPPINGS.items():
            if team_name in variants or team_name == canonical_name:
                return canonical_name
        return team_name

    def get_team_features(self, home_team: str, away_team: str, home_rest_days: int = 7, away_rest_days: int = 7) -> Optional[Dict]:
        """Get comprehensive features for two teams"""
        if not self.team_stats:
            return None
        
        # Normalize team names
        home_team = self.normalize_team_name(home_team)
        away_team = self.normalize_team_name(away_team)
        
        home_stats = self.team_stats.get(home_team)
        away_stats = self.team_stats.get(away_team)
        
        if not home_stats or not away_stats:
            logger.warning(f"Stats not found for teams: {home_team}, {away_team}")
            return None
        
        # Calculate head-to-head
        h2h = self.calculate_head_to_head(home_team, away_team)
        
        # Build features
        features = {
            'home_elo': home_stats.get('elo', 1500),
            'away_elo': away_stats.get('elo', 1500),
            'home_form': home_stats.get('form', 0),
            'away_form': away_stats.get('form', 0),
            'home_avg_scored': home_stats.get('avg_scored', 1.5),
            'home_avg_conceded': home_stats.get('avg_conceded', 1.5),
            'away_avg_scored': away_stats.get('avg_scored', 1.5),
            'away_avg_conceded': away_stats.get('avg_conceded', 1.5),
            'h2h_home_wins': h2h['h2h_home_wins'],
            'h2h_draws': h2h['h2h_draws'],
            'h2h_away_wins': h2h['h2h_away_wins'],
            'home_rest_days': home_rest_days,
            'away_rest_days': away_rest_days
        }
        
        # Calculate derived features
        features.update({
            'elo_diff': features['home_elo'] - features['away_elo'],
            'form_diff': features['home_form'] - features['away_form'],
            'attack_strength_diff': features['home_avg_scored'] - features['away_avg_conceded'],
            'rest_advantage': features['home_rest_days'] - features['away_rest_days']
        })
        
        return features

    def calculate_head_to_head(self, home_team: str, away_team: str) -> Dict:
        """Calculate head-to-head statistics between two teams"""
        if self.historical_data.empty:
            return {'h2h_home_wins': 0, 'h2h_draws': 0, 'h2h_away_wins': 0}
        
        h2h_matches = self.historical_data[
            ((self.historical_data['home_team'] == home_team) & (self.historical_data['away_team'] == away_team)) |
            ((self.historical_data['home_team'] == away_team) & (self.historical_data['away_team'] == home_team))
        ].tail(10)  # Last 10 encounters
        
        home_wins = 0
        draws = 0
        away_wins = 0
        
        for _, match in h2h_matches.iterrows():
            if match['home_team'] == home_team:
                if match['result'] == 2: home_wins += 1
                elif match['result'] == 1: draws += 1
                else: away_wins += 1
            else:
                if match['result'] == 0: home_wins += 1
                elif match['result'] == 1: draws += 1
                else: away_wins += 1
        
        return {
            'h2h_home_wins': home_wins,
            'h2h_draws': draws,
            'h2h_away_wins': away_wins
        }

    def predict(self, features: Dict) -> Tuple[int, Dict[str, float], float]:
        """Make prediction"""
        df = pd.DataFrame([features])
        
        # Ensure correct feature order
        if self.feature_names:
            df = df.reindex(columns=self.feature_names, fill_value=0)
        
        prediction_code = int(self.model.predict(df)[0])
        
        # Calculate probabilities
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(df)[0]
            probabilities = {
                "away_win": float(proba[0]),
                "draw": float(proba[1]),
                "home_win": float(proba[2])
            }
            confidence = float(max(proba))
        else:
            probabilities = {
                "away_win": 1.0 if prediction_code == 0 else 0.0,
                "draw": 1.0 if prediction_code == 1 else 0.0,
                "home_win": 1.0 if prediction_code == 2 else 0.0
            }
            confidence = 1.0
        
        return prediction_code, probabilities, confidence

    def get_available_teams(self) -> List[str]:
        """Get list of available teams"""
        return list(self.team_stats.keys()) if self.team_stats else []

    def predict_upcoming_matches(self) -> List[Dict]:
        """Predict all upcoming matches"""
        if self.upcoming_matches.empty:
            return []
        
        predictions = []
        for _, match in self.upcoming_matches.iterrows():
            try:
                features = self.get_team_features(
                    match['home_team'],
                    match['away_team']
                )
                
                if features:
                    prediction_code, probabilities, confidence = self.predict(features)
                    outcome_map = {0: "Away Win", 1: "Draw", 2: "Home Win"}
                    
                    predictions.append({
                        "date": match['date'],
                        "home_team": match['home_team'],
                        "away_team": match['away_team'],
                        "competition": match.get('competition', 'Unknown'),
                        "prediction": outcome_map[prediction_code],
                        "prediction_code": prediction_code,
                        "probabilities": probabilities,
                        "confidence": confidence
                    })
                    
            except Exception as e:
                logger.error(f"Error predicting match {match['home_team']} vs {match['away_team']}: {e}")
                continue
        
        return predictions

# Initialize model manager
model_manager = ModelManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle (startup and shutdown)"""
    # Startup
    logger.info("Starting up...")
    if not model_manager.load_best_model():
        logger.error("Failed to load model on startup")
    yield
    # Shutdown
    logger.info("Shutting down...")

app = FastAPI(
    title="Football Match Predictor", 
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Helper functions
def calculate_derived_features(input_data: MatchInput) -> Dict:
    """Calculate derived features"""
    features = input_data.model_dump()
    
    # Calculate derived features
    derived_features = {
        'elo_diff': features['home_elo'] - features['away_elo'],
        'form_diff': features['home_form'] - features['away_form'],
        'attack_strength_diff': features['home_avg_scored'] - features['away_avg_conceded'],
        'rest_advantage': features['home_rest_days'] - features['away_rest_days']
    }
    
    features.update(derived_features)
    return features

# Root and health endpoints
@app.get("/")
async def root():
    """API info"""
    return {
        "service": "Football Match Predictor",
        "status": "ready" if model_manager.model else "no_model",
        "model": model_manager.model_name,
        "features_loaded": len(model_manager.feature_names) if model_manager.feature_names else 0,
        "teams_loaded": len(model_manager.team_stats) if model_manager.team_stats else 0,
        "upcoming_matches": len(model_manager.upcoming_matches) if model_manager.upcoming_matches is not None else 0
    }

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy" if model_manager.model else "unhealthy", 
        "model_loaded": model_manager.model is not None,
        "model_name": model_manager.model_name,
        "teams_loaded": len(model_manager.team_stats) if model_manager.team_stats else 0,
        "upcoming_matches_loaded": len(model_manager.upcoming_matches) if model_manager.upcoming_matches is not None else 0
    }

# Existing manual prediction endpoints
@app.get("/model/info")
async def model_info():
    """Model information"""
    if not model_manager.model:
        raise HTTPException(status_code=503, detail="No model loaded")
    
    info = {
        "model_name": model_manager.model_name,
        "model_type": type(model_manager.model).__name__,
        "n_features": len(model_manager.feature_names) if model_manager.feature_names else 0,
        "features": model_manager.feature_names if model_manager.feature_names else []
    }
    
    # Add model-specific info
    if hasattr(model_manager.model, 'n_estimators'):
        info['n_estimators'] = model_manager.model.n_estimators
    if hasattr(model_manager.model, 'feature_importances_'):
        info['has_feature_importance'] = True
    
    return info

@app.post("/predict", response_model=PredictionResponse)
async def predict(input_data: MatchInput):
    """Manual prediction endpoint"""
    if not model_manager.model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Calculate features and predict
        full_features = calculate_derived_features(input_data)
        prediction_code, probabilities, confidence = model_manager.predict(full_features)
        
        outcome_map = {0: "Away Win", 1: "Draw", 2: "Home Win"}
        
        return PredictionResponse(
            prediction=outcome_map[prediction_code],
            prediction_code=prediction_code,
            probabilities=probabilities,
            confidence=confidence,
            model_used=model_manager.model_name,
            features_used=full_features
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Team-based prediction endpoints
@app.get("/teams")
async def get_available_teams():
    """Get list of available teams"""
    teams = model_manager.get_available_teams()
    return {
        "count": len(teams),
        "teams": sorted(teams)
    }

@app.get("/teams/{team_name}/stats")
async def get_team_stats(team_name: str):
    """Get statistics for a specific team"""
    if not model_manager.team_stats:
        raise HTTPException(status_code=503, detail="Team statistics not loaded")
    
    # Normalize team name
    normalized_name = model_manager.normalize_team_name(team_name)
    
    stats = model_manager.team_stats.get(normalized_name)
    if not stats:
        raise HTTPException(status_code=404, detail=f"Team '{team_name}' not found. Available teams: {list(model_manager.team_stats.keys())[:10]}...")
    
    return {
        "team": normalized_name,
        "stats": stats
    }

@app.post("/predict/teams", response_model=TeamPredictionResponse)
async def predict_by_teams(request: TeamPredictionRequest):
    """Predict match outcome by team names"""
    if not model_manager.model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Get features for the teams
        features = model_manager.get_team_features(
            request.home_team, 
            request.away_team,
            request.home_rest_days,
            request.away_rest_days
        )
        
        if not features:
            available_teams = model_manager.get_available_teams()
            raise HTTPException(
                status_code=404, 
                detail=f"Could not generate features for teams: {request.home_team} vs {request.away_team}. Available teams: {available_teams[:10]}..."
            )
        
        # Make prediction
        prediction_code, probabilities, confidence = model_manager.predict(features)
        
        outcome_map = {0: "Away Win", 1: "Draw", 2: "Home Win"}
        
        return TeamPredictionResponse(
            home_team=request.home_team,
            away_team=request.away_team,
            prediction=outcome_map[prediction_code],
            prediction_code=prediction_code,
            probabilities=probabilities,
            confidence=confidence,
            model_used=model_manager.model_name,
            features_used={k: round(v, 3) if isinstance(v, float) else v for k, v in features.items()}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Team prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Upcoming matches endpoints
@app.get("/upcoming-matches")
async def get_upcoming_matches():
    """Get upcoming matches"""
    if model_manager.upcoming_matches is None or model_manager.upcoming_matches.empty:
        raise HTTPException(status_code=404, detail="No upcoming matches available")
    
    matches = model_manager.upcoming_matches.to_dict('records')
    return {
        "count": len(matches),
        "matches": matches
    }

@app.get("/predict/upcoming")
async def predict_upcoming_matches():
    """Predict all upcoming matches"""
    if not model_manager.model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        predictions = model_manager.predict_upcoming_matches()
        return {
            "count": len(predictions),
            "predictions": predictions
        }
    except Exception as e:
        logger.error(f"Upcoming matches prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/upcoming-custom")
async def predict_custom_upcoming_matches(matches: List[UpcomingMatch]):
    """Predict custom upcoming matches"""
    if not model_manager.model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    predictions = []
    
    for match in matches:
        try:
            features = model_manager.get_team_features(
                match.home_team,
                match.away_team
            )
            
            if features:
                prediction_code, probabilities, confidence = model_manager.predict(features)
                outcome_map = {0: "Away Win", 1: "Draw", 2: "Home Win"}
                
                predictions.append(UpcomingPredictionResponse(
                    date=match.date,
                    home_team=match.home_team,
                    away_team=match.away_team,
                    competition=match.competition,
                    prediction=outcome_map[prediction_code],
                    prediction_code=prediction_code,
                    probabilities=probabilities,
                    confidence=confidence
                ))
            else:
                predictions.append({
                    "date": match.date,
                    "home_team": match.home_team,
                    "away_team": match.away_team,
                    "competition": match.competition,
                    "error": "Could not generate features",
                    "success": False
                })
                
        except Exception as e:
            predictions.append({
                "date": match.date,
                "home_team": match.home_team,
                "away_team": match.away_team,
                "competition": match.competition,
                "error": str(e),
                "success": False
            })
    
    return {
        "count": len(predictions),
        "successful": len([p for p in predictions if isinstance(p, UpcomingPredictionResponse)]),
        "predictions": predictions
    }

# Feature importance endpoint
@app.get("/model/feature_importance")
async def get_feature_importance():
    """Feature importance"""
    if not model_manager.model:
        raise HTTPException(status_code=503, detail="No model loaded")
    if not hasattr(model_manager.model, 'feature_importances_'):
        raise HTTPException(status_code=400, detail="Feature importance not supported")
    if not model_manager.feature_names:
        raise HTTPException(status_code=500, detail="Feature names unavailable")
    
    importance = model_manager.model.feature_importances_
    feature_importance = [
        {"feature": name, "importance": float(imp)}
        for name, imp in zip(model_manager.feature_names, importance)
    ]
    feature_importance.sort(key=lambda x: x['importance'], reverse=True)
    
    return {
        "model": model_manager.model_name, 
        "feature_importance": feature_importance
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host=os.getenv("API_HOST", "127.0.0.1"), 
        port=int(os.getenv("API_PORT", "8000")),
        log_level="info"
    )