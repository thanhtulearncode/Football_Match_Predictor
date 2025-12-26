import pandas as pd
import joblib
import logging
import pandera.pandas as pa
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, log_loss
from config.settings import settings
from src.pipeline.validation import TrainingSchema
from src.pipeline.processing import preprocess_time_features, get_fitted_encoders, apply_encoders

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

logger = logging.getLogger(__name__)

class ModelTrainer:

    def __init__(self, data_path=None):
        self.data_path = data_path or settings.MATCHES_FINAL_PATH
        self.models = self._define_models()

    def _define_models(self):
        """Define available models."""
        models = {
            "RandomForest": RandomForestClassifier(
                # TUNED PARAMETERS
                n_estimators=300,
                min_samples_split=2,
                min_samples_leaf=4,
                max_features='log2',
                max_depth=None,
                random_state=42
            ),
        }
        
        if XGBClassifier:
            models["XGBoost"] = XGBClassifier(
                # TUNED PARAMETERS
                n_estimators=200,
                learning_rate=0.01,
                max_depth=4,
                subsample=0.6,
                colsample_bytree=0.8,
                gamma=0,
                eval_metric='mlogloss',
                random_state=42
            )
        return models

    def train(self):
        logger.info("Loading training data...")
        if not self.data_path.exists():
            raise FileNotFoundError(f"Training data not found at {self.data_path}") 
        df = pd.read_csv(self.data_path)
        # Centralized Time Processing
        df = preprocess_time_features(df)
        # Fit and Save Encoders
        logger.info("Fitting and saving encoders...")
        encoders = get_fitted_encoders(df)
        df = apply_encoders(df, encoders)
        joblib.dump(encoders, settings.ENCODERS_PATH)
        required_cols = settings.PREDICTORS + ["target"]
        df = df.dropna(subset=required_cols)
        try:
            TrainingSchema.validate(df, lazy=True)
        except pa.errors.SchemaErrors as e:
            logger.error(f"Training data schema invalid:\n{e.failure_cases}")
            return
        # Train-Test Split
        split_date = settings.SPLIT_DATE
        train = df[df["date"] < split_date]
        test = df[df["date"] >= split_date]
        if len(train) < 50:
            logger.error("Not enough training data. Aborting.")
            return
        logger.info(f"Training set: {len(train)} | Test set: {len(test)}")
        best_model_data = {"name": None, "obj": None, "acc": 0.0, "log_loss": float('inf')}

        for name, base_model in self.models.items():
            try:
                logger.info(f"Training {name} (Calibrated)...")
                # CalibratedClassifierCV using the tuned base_model
                calibrated_model = CalibratedClassifierCV(
                    estimator=base_model,
                    method='isotonic' if len(train) > 1000 else 'sigmoid',
                    cv=3 
                )
                
                calibrated_model.fit(train[settings.PREDICTORS], train["target"])
                # Predictions
                preds = calibrated_model.predict(test[settings.PREDICTORS])
                probs = calibrated_model.predict_proba(test[settings.PREDICTORS])
                # Metrics
                acc = accuracy_score(test["target"], preds)
                prec = precision_score(test["target"], preds, average='weighted', zero_division=0)
                ll = log_loss(test["target"], probs)
                
                logger.info(f"Results for {name}:")
                logger.info(f"  - Accuracy:  {acc:.4f}")
                logger.info(f"  - Precision: {prec:.4f}")
                logger.info(f"  - Log Loss:  {ll:.4f} (Lower is better)")
                
                # Selection logic: Prioritizing Accuracy, using Log Loss as tie-breaker
                if acc > best_model_data["acc"]:
                    best_model_data = {
                        "name": name, 
                        "obj": calibrated_model, 
                        "acc": acc,
                        "log_loss": ll
                    }
                elif acc == best_model_data["acc"] and ll < best_model_data["log_loss"]:
                    # If accuracy is tied, pick the one with better (lower) log loss
                    best_model_data = {
                        "name": name, 
                        "obj": calibrated_model, 
                        "acc": acc,
                        "log_loss": ll
                    }
                    
            except Exception as e:
                logger.error(f"Failed to train {name}: {e}", exc_info=True)

        if not best_model_data["obj"]:
            logger.error("All models failed.")
            return

        if best_model_data["acc"] < settings.MIN_ACCURACY_THRESHOLD:
            logger.warning(
                f"Best model {best_model_data['name']} "
                f"(Accuracy: {best_model_data['acc']:.2f}) failed quality gate."
            )

        # Versioned Saving
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        versioned_filename = f"model_{best_model_data['name']}_{timestamp}.pkl"
        versioned_path = settings.MODELS_DIR / versioned_filename
        # Save versioned copy
        joblib.dump(best_model_data["obj"], versioned_path)
        # Update 'latest' pointer
        latest_path = settings.MODEL_PATH
        joblib.dump(best_model_data["obj"], latest_path)
        logger.info(f"Saved new best model: {versioned_filename}")
        logger.info(f"Updated production model at: {latest_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ModelTrainer().train()