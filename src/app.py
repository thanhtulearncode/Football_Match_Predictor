"""FastAPI app for football match predictions"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from pathlib import Path
from typing import Dict, List

app = FastAPI(title="Football Match Predictor", version="2.0.0")
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
model = None
model_name = None
feature_names = None


class MatchFeatures(BaseModel):
    """Match prediction input features"""
    home_elo: float = Field(..., ge=1000, le=2500)
    away_elo: float = Field(..., ge=1000, le=2500)
    elo_diff: float = Field(...)
    home_form: float = Field(..., ge=0, le=15)
    away_form: float = Field(..., ge=0, le=15)
    form_diff: float = Field(...)
    home_avg_scored: float = Field(..., ge=0)
    home_avg_conceded: float = Field(..., ge=0)
    away_avg_scored: float = Field(..., ge=0)
    away_avg_conceded: float = Field(..., ge=0)
    attack_strength_diff: float = Field(...)
    h2h_home_wins: int = Field(..., ge=0)
    h2h_draws: int = Field(..., ge=0)
    h2h_away_wins: int = Field(..., ge=0)
    home_rest_days: int = Field(..., ge=0, le=30)
    away_rest_days: int = Field(..., ge=0, le=30)
    rest_advantage: int = Field(...)


class PredictionResponse(BaseModel):
    """Prediction response"""
    prediction: str
    prediction_code: int
    probabilities: Dict[str, float]
    confidence: float
    model_used: str

@app.on_event("startup")
async def load_model():
    """Load best model on startup"""
    global model, model_name, feature_names
    comparison_path = MODELS_DIR / "model_comparison.csv"
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path, index_col=0)
        best_model_name = comparison['test_accuracy'].idxmax()
        model_path = MODELS_DIR / f"{best_model_name}_model.pkl"
    else:
        for name in ['XGBoost', 'LightGBM', 'RandomForest', 'GradientBoosting']:
            model_path = MODELS_DIR / f"{name}_model.pkl"
            if model_path.exists():
                best_model_name = name
                break
        else:
            model_path = None
    if model_path and model_path.exists():
        try:
            model = joblib.load(model_path)
            model_name = best_model_name
            if hasattr(model, 'feature_names_in_'):
                feature_names = model.feature_names_in_.tolist()
            print(f"Loaded: {model_name}")
        except Exception as e:
            print(f"Error: {e}")
            model = None
    else:
        print("No model found. Run train.py first.")


@app.get("/")
async def root():
    """API info"""
    return {
        "service": "Football Match Predictor",
        "status": "ready" if model else "no_model",
        "model": model_name
    }

@app.get("/health")
async def health_check():
    """Health check"""
    return {"status": "healthy" if model else "unhealthy", "model_loaded": model is not None}

@app.get("/model/info")
async def model_info():
    """Model information"""
    if not model:
        raise HTTPException(status_code=503, detail="No model loaded")
    info = {
        "model_name": model_name,
        "model_type": type(model).__name__,
        "n_features": len(feature_names) if feature_names else 0
    }
    if hasattr(model, 'n_estimators'):
        info['n_estimators'] = model.n_estimators
    return info


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: MatchFeatures):
    """Predict match outcome"""
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        df = pd.DataFrame([features.dict()])
        if feature_names:
            df = df[feature_names]
        prediction_code = int(model.predict(df)[0])
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(df)[0]
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
        outcome_map = {0: "Away Win", 1: "Draw", 2: "Home Win"}
        return PredictionResponse(
            prediction=outcome_map[prediction_code],
            prediction_code=prediction_code,
            probabilities=probabilities,
            confidence=confidence,
            model_used=model_name
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(matches: List[MatchFeatures]):
    """Batch predictions"""
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    predictions = []
    for match_features in matches:
        try:
            pred = await predict(match_features)
            predictions.append(pred.dict())
        except Exception as e:
            predictions.append({"error": str(e)})
    return {"count": len(predictions), "predictions": predictions}

@app.get("/model/feature_importance")
async def get_feature_importance():
    """Feature importance"""
    if not model:
        raise HTTPException(status_code=503, detail="No model loaded")
    if not hasattr(model, 'feature_importances_'):
        raise HTTPException(status_code=400, detail="Not supported")
    if not feature_names:
        raise HTTPException(status_code=500, detail="Feature names unavailable")
    importance = model.feature_importances_
    feature_importance = [
        {"feature": name, "importance": float(imp)}
        for name, imp in zip(feature_names, importance)
    ]
    feature_importance.sort(key=lambda x: x['importance'], reverse=True)
    return {"model": model_name, "feature_importance": feature_importance}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)