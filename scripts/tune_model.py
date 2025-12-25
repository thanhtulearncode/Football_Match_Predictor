import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, log_loss, accuracy_score

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.settings import settings
from src.pipeline.processing import preprocess_time_features, get_fitted_encoders, apply_encoders

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def load_and_prep_data():
    """Load data and apply the EXACT same preprocessing as the training pipeline."""
    data_path = settings.MATCHES_FINAL_PATH
    if not data_path.exists():
        raise FileNotFoundError(f"Data not found at {data_path}. Run the pipeline first.")
    df = pd.read_csv(data_path)
    # 1. Time Features
    df = preprocess_time_features(df)
    # 2. Encoding
    encoders = get_fitted_encoders(df)
    df = apply_encoders(df, encoders)
    # 3. Drop NAs in predictors
    required_cols = settings.PREDICTORS + ["target"]
    df = df.dropna(subset=required_cols)
    return df

def tune_models():
    df = load_and_prep_data()
    # Sort by date for TimeSeriesSplit
    df = df.sort_values("date").reset_index(drop=True)
    X = df[settings.PREDICTORS]
    y = df["target"]
    # TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=3)
    # --- Define Parameter Grids ---
    # Random Forest Grid
    rf_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    # XGBoost Grid 
    xgb_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6, 8],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2]
    }
    models_to_tune = [
        ("RandomForest", RandomForestClassifier(random_state=42), rf_grid),
    ]

    try:
        from xgboost import XGBClassifier
        models_to_tune.append(
            ("XGBoost", XGBClassifier(eval_metric='mlogloss', random_state=42), xgb_grid)
        )
    except ImportError:
        logger.warning("XGBoost not installed. Skipping.")

    logger.info(f"Starting tuning on {len(df)} rows with {len(settings.PREDICTORS)} features...")

    for name, model, grid in models_to_tune:
        logger.info(f"--- Tuning {name} ---")
        # Optimize for Log Loss
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=grid,
            n_iter=20, 
            scoring='neg_log_loss', # Minimize Log Loss
            cv=tscv,
            verbose=1,
            n_jobs=-1,
            random_state=42
        )
        
        search.fit(X, y)
        
        best_score_ll = -search.best_score_
        logger.info(f"Best {name} Log Loss: {best_score_ll:.4f}")
        logger.info(f"Best Params: {search.best_params_}")
        # Validation: Check Accuracy of this best model
        best_model = search.best_estimator_
        # Simple test on last 20% of data manually to confirm
        split_idx = int(len(df) * 0.8)
        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]
        preds = best_model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        logger.info(f"Validation Accuracy on most recent data: {acc:.4f}")
        logger.info("-" * 30)

if __name__ == "__main__":
    tune_models()