"""FastAPI app for football match predictions"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator
import joblib
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from contextlib import asynccontextmanager
import os

# Import config
from src.config import MODELS_DIR, FEATURES

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MatchInput(BaseModel):
    """Input model for match prediction API with validation"""
    # Elo ratings (typically 1000-2500)
    home_elo: float = Field(..., ge=1000, le=2500)
    away_elo: float = Field(..., ge=1000, le=2500)
    # Form points (points from last N matches, typically 0-15)
    home_form: float = Field(..., ge=0, le=15)
    away_form: float = Field(..., ge=0, le=15)
    # Average goals scored/conceded
    home_avg_scored: float = Field(..., ge=0)
    home_avg_conceded: float = Field(..., ge=0)
    away_avg_scored: float = Field(..., ge=0)
    away_avg_conceded: float = Field(..., ge=0)
    # Head-to-head history
    h2h_home_wins: int = Field(..., ge=0)
    h2h_draws: int = Field(..., ge=0)
    h2h_away_wins: int = Field(..., ge=0)
    # Rest days (0-30 days)
    home_rest_days: int = Field(..., ge=0, le=30)
    away_rest_days: int = Field(..., ge=0, le=30)

    @model_validator(mode='after')
    def validate_h2h_totals(self):
        """Validate that head-to-head statistics are reasonable"""
        total = self.h2h_home_wins + self.h2h_draws + self.h2h_away_wins
        if total > 20:  # Reasonable limit for historical matches
            raise ValueError('Total H2H matches too high')
        return self

class PredictionResponse(BaseModel):
    """Prediction response"""
    prediction: str
    prediction_code: int
    probabilities: Dict[str, float]
    confidence: float
    model_used: str

class ModelManager:
    """Singleton pattern model manager for loading and using ML models"""
    _instance = None
    
    def __new__(cls):
        # Singleton: ensure only one instance exists
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model = None
            cls._instance.model_name = None
            cls._instance.feature_names = None
        return cls._instance
    
    def load_best_model(self) -> bool:
        """Load the best performing model from comparison file"""
        try:
            # Try to load from comparison file first
            comparison_path = MODELS_DIR / "model_comparison.csv"
            if comparison_path.exists():
                comparison = pd.read_csv(comparison_path, index_col=0)
                # Find model with highest test accuracy
                best_model_name = comparison['test_accuracy'].idxmax()
                model_path = MODELS_DIR / f"{best_model_name}_model.pkl"
                logger.info(f"Best model from comparison: {best_model_name}")
            else:
                # Fallback: try to load first available model
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

            # Load the model from file
            if model_path.exists():
                self.model = joblib.load(model_path)
                self.model_name = best_model_name
                
                # Get feature names from model or use default list
                if hasattr(self.model, 'feature_names_in_'):
                    self.feature_names = self.model.feature_names_in_.tolist()
                else:
                    self.feature_names = FEATURES
                
                logger.info(f"Successfully loaded: {self.model_name} with {len(self.feature_names)} features")
                return True
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
        
        return False

    def predict(self, features: Dict) -> Tuple[int, Dict[str, float], float]:
        """Make prediction from feature dictionary"""
        # Convert features to DataFrame
        df = pd.DataFrame([features])
        
        # Ensure features are in correct order (required by sklearn)
        if self.feature_names:
            df = df.reindex(columns=self.feature_names, fill_value=0)
        
        # Get prediction code: 0=away win, 1=draw, 2=home win
        prediction_code = int(self.model.predict(df)[0])
        
        # Calculate class probabilities if available
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(df)[0]
            probabilities = {
                "away_win": float(proba[0]),
                "draw": float(proba[1]),
                "home_win": float(proba[2])
            }
            # Confidence is the maximum probability
            confidence = float(max(proba))
        else:
            # Hard prediction if probabilities not available
            probabilities = {
                "away_win": 1.0 if prediction_code == 0 else 0.0,
                "draw": 1.0 if prediction_code == 1 else 0.0,
                "home_win": 1.0 if prediction_code == 2 else 0.0
            }
            confidence = 1.0
        
        return prediction_code, probabilities, confidence

def calculate_derived_features(input_data: MatchInput) -> Dict:
    """Calculate derived features from input data"""
    # Convert Pydantic model to dictionary
    features = input_data.model_dump()
    
    # Calculate difference features (team advantages)
    derived_features = {
        'elo_diff': features['home_elo'] - features['away_elo'],  # Elo rating difference
        'form_diff': features['home_form'] - features['away_form'],  # Form difference
        'attack_strength_diff': features['home_avg_scored'] - features['away_avg_conceded'],  # Attack vs defense
        'rest_advantage': features['home_rest_days'] - features['away_rest_days']  # Rest days difference
    }
    
    # Merge derived features with original features
    features.update(derived_features)
    return features

# Initialize model manager
model_manager = ModelManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle (startup and shutdown)"""
    # Startup: load model when application starts
    logger.info("Starting up...")
    if not model_manager.load_best_model():
        logger.error("Failed to load model on startup")
    yield
    # Shutdown: cleanup when application stops
    logger.info("Shutting down...")

app = FastAPI(
    title="Football Match Predictor", 
    version="2.1.0",
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

@app.get("/")
async def root():
    """API info"""
    return {
        "service": "Football Match Predictor",
        "status": "ready" if model_manager.model else "no_model",
        "model": model_manager.model_name,
        "features_loaded": len(model_manager.feature_names) if model_manager.feature_names else 0
    }

@app.get("/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy" if model_manager.model else "unhealthy", 
        "model_loaded": model_manager.model is not None,
        "model_name": model_manager.model_name
    }

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
    """Predict match outcome from input features"""
    # Check if model is loaded
    if not model_manager.model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Calculate derived features and make prediction
        full_features = calculate_derived_features(input_data)
        prediction_code, probabilities, confidence = model_manager.predict(full_features)
        
        # Map prediction code to outcome name
        outcome_map = {0: "Away Win", 1: "Draw", 2: "Home Win"}
        
        return PredictionResponse(
            prediction=outcome_map[prediction_code],
            prediction_code=prediction_code,
            probabilities=probabilities,
            confidence=confidence,
            model_used=model_manager.model_name
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/batch")
async def predict_batch(matches: List[MatchInput]):
    """Predict outcomes for multiple matches in batch"""
    if not model_manager.model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    predictions = []
    # Process each match in the batch
    for i, match_features in enumerate(matches):
        try:
            # Calculate features and predict
            full_features = calculate_derived_features(match_features)
            prediction_code, probabilities, confidence = model_manager.predict(full_features)
            
            outcome_map = {0: "Away Win", 1: "Draw", 2: "Home Win"}
            
            predictions.append({
                "prediction": outcome_map[prediction_code],
                "prediction_code": prediction_code,
                "probabilities": probabilities,
                "confidence": confidence,
                "model_used": model_manager.model_name
            })
        except Exception as e:
            # Handle errors for individual matches without failing entire batch
            predictions.append({
                "error": f"Match {i}: {str(e)}",
                "success": False
            })
    
    return {
        "count": len(predictions), 
        "successful": len([p for p in predictions if "error" not in p]),
        "predictions": predictions
    }

@app.get("/model/feature_importance")
async def get_feature_importance():
    """Get feature importance scores from the loaded model"""
    if not model_manager.model:
        raise HTTPException(status_code=503, detail="No model loaded")
    if not hasattr(model_manager.model, 'feature_importances_'):
        raise HTTPException(status_code=400, detail="Feature importance not supported")
    if not model_manager.feature_names:
        raise HTTPException(status_code=500, detail="Feature names unavailable")
    
    # Get feature importance scores
    importance = model_manager.model.feature_importances_
    # Create list of feature-importance pairs
    feature_importance = [
        {"feature": name, "importance": float(imp)}
        for name, imp in zip(model_manager.feature_names, importance)
    ]
    # Sort by importance (highest first)
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