import pandas as pd
import joblib
import logging
import pandera as pa
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score
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
                n_estimators=100, min_samples_split=10, random_state=42
            ),
        }
        if XGBClassifier:
            models["XGBoost"] = XGBClassifier(
                n_estimators=100, learning_rate=0.05, max_depth=4, 
                eval_metric='mlogloss', random_state=42
            )
        return models

    def train(self):
        logger.info("Loading training data...")
        if not self.data_path.exists():
            raise FileNotFoundError(f"Training data not found at {self.data_path}") 
        df = pd.read_csv(self.data_path)
        # Centralized Time Processing ---
        df = preprocess_time_features(df)
        # Fit and Save Encoders (Crucial for skew prevention) ---
        logger.info("Fitting and saving encoders...")
        encoders = get_fitted_encoders(df)
        df = apply_encoders(df, encoders)
        # Save encoders
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
        best_model_data = {"name": None, "obj": None, "acc": 0.0}

        for name, model in self.models.items():
            try:
                model.fit(train[settings.PREDICTORS], train["target"])
                preds = model.predict(test[settings.PREDICTORS])
                
                acc = accuracy_score(test["target"], preds)
                prec = precision_score(test["target"], preds, average='weighted', zero_division=0)
                
                logger.info(f"{name}: Accuracy={acc:.2f}, Precision={prec:.2f}")
                
                if acc > best_model_data["acc"]:
                    best_model_data = {"name": name, "obj": model, "acc": acc}
            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")

        if not best_model_data["obj"]:
            logger.error("All models failed.")
            return

        if best_model_data["acc"] < settings.MIN_ACCURACY_THRESHOLD:
            logger.warning(
                f"Best model {best_model_data['name']} "
                f"({best_model_data['acc']:.2f}) failed quality gate."
            )

        output_path = settings.MODEL_PATH
        joblib.dump(best_model_data["obj"], output_path)
        logger.info(f"Saved {best_model_data['name']} to {output_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ModelTrainer().train()