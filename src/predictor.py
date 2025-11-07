"""Enhanced predictor module for dashboard compatibility"""
import joblib
import pandas as pd
from pathlib import Path
import logging
from typing import Dict, Optional
from src.config import MODELS_DIR, MODEL_COMPARISON_FILE, FEATURES

logger = logging.getLogger(__name__)

class MatchPredictor:
    """Predictor class for loading models and making match predictions"""
    def __init__(self):
        self.model = None  # Trained ML model
        self.model_name = None  # Name of the loaded model
        self.feature_names = None  # List of feature names expected by model
        self._accuracy = None  # Model accuracy score
    
    def load_best_model(self) -> bool:
        """Load the best performing model from comparison file"""
        try:
            # Check if comparison file exists
            if not MODEL_COMPARISON_FILE.exists():
                return self._load_fallback_model()
            
            # Read model comparison results
            comparison = pd.read_csv(MODEL_COMPARISON_FILE, index_col=0)
            # Find model with highest test accuracy
            best_model_name = comparison['test_accuracy'].idxmax()
            model_path = MODELS_DIR / f"{best_model_name}_model.pkl"
            
            # Verify model file exists
            if not model_path.exists():
                logger.warning(f"Best model not found: {model_path}, trying fallback")
                return self._load_fallback_model()
            
            # Load the model
            self.model = joblib.load(model_path)
            self.model_name = best_model_name
            self._accuracy = comparison.loc[best_model_name, 'test_accuracy']
            
            # Get feature names from model or use default list
            if hasattr(self.model, 'feature_names_in_'):
                self.feature_names = self.model.feature_names_in_.tolist()
            else:
                self.feature_names = FEATURES
            
            logger.info(f"Successfully loaded: {self.model_name} (Accuracy: {self._accuracy:.4f})")
            return True
            
        except Exception as e:
            logger.error(f"Error loading best model: {e}")
            return self._load_fallback_model()
    
    def _load_fallback_model(self) -> bool:
        """Load first available model as fallback if best model is unavailable"""
        # Priority order of models to try
        available_models = ['XGBoost', 'LightGBM', 'RandomForest', 'GradientBoosting']
        
        # Try each model in priority order
        for name in available_models:
            model_path = MODELS_DIR / f"{name}_model.pkl"
            if model_path.exists():
                try:
                    # Load the model
                    self.model = joblib.load(model_path)
                    self.model_name = name
                    
                    # Set feature names
                    if hasattr(self.model, 'feature_names_in_'):
                        self.feature_names = self.model.feature_names_in_.tolist()
                    else:
                        self.feature_names = FEATURES
                    
                    logger.info(f"Loaded fallback model: {self.model_name}")
                    return True
                    
                except Exception as e:
                    logger.error(f"Error loading fallback model {name}: {e}")
                    continue
        
        logger.error("No model files found!")
        return False
    
    def get_accuracy(self) -> Optional[float]:
        """Get model accuracy score if available"""
        return self._accuracy
    
    def predict(self, features_dict: dict) -> Dict:
        """Make match prediction from feature dictionary"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_best_model() first.")
        
        # Convert features to DataFrame
        df = pd.DataFrame([features_dict])
        # Ensure features are in correct order (required by sklearn models)
        if self.feature_names:
            df = df.reindex(columns=self.feature_names, fill_value=0)
        
        # Get prediction code: 0=away win, 1=draw, 2=home win
        prediction_code = int(self.model.predict(df)[0])
        
        # Calculate class probabilities if model supports it
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(df)[0]
            probabilities = {
                "away_win": float(proba[0]),  # Probability of away win
                "draw": float(proba[1]),      # Probability of draw
                "home_win": float(proba[2])   # Probability of home win
            }
            # Confidence is the maximum probability
            confidence = float(max(proba))
        else:
            # If probabilities not available, use hard prediction
            probabilities = {
                "away_win": 1.0 if prediction_code == 0 else 0.0,
                "draw": 1.0 if prediction_code == 1 else 0.0,
                "home_win": 1.0 if prediction_code == 2 else 0.0
            }
            confidence = 1.0
        
        # Map prediction code to outcome name
        outcome_map = {0: "Away Win", 1: "Draw", 2: "Home Win"}
        
        return {
            "prediction": outcome_map[prediction_code],
            "prediction_code": prediction_code,
            "probabilities": probabilities,
            "confidence": confidence,
            "model_used": self.model_name
        }

# Singleton instance
predictor = MatchPredictor()