"""Train football prediction models"""
import pandas as pd
import numpy as np
from pathlib import Path
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, log_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

import joblib
import logging
from typing import Dict, Tuple, Any, List
from .config import PROCESSED_DIR as PROCESSED, MODELS_DIR as MODELS, RANDOM_STATE, TARGET_COLUMN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dynamic imports with fallbacks (optional libraries)
MODEL_IMPORTS = {}
try:
    from xgboost import XGBClassifier
    MODEL_IMPORTS['XGBoost'] = XGBClassifier
except ImportError:
    logger.warning("XGBoost not available")

try:
    from lightgbm import LGBMClassifier
    MODEL_IMPORTS['LightGBM'] = LGBMClassifier
except ImportError:
    logger.warning("LightGBM not available")

try:
    from catboost import CatBoostClassifier
    MODEL_IMPORTS['CatBoost'] = CatBoostClassifier
except ImportError:
    logger.warning("CatBoost not available")

class ModelTrainer:
    """Train and compare multiple ML models for football prediction"""
    def __init__(self):
        # Configuration for each model type (class and hyperparameters)
        self.models_config = {
            'RandomForest': {
                'class': RandomForestClassifier,
                'params': {
                    'n_estimators': 200,      # Number of trees
                    'max_depth': 10,          # Maximum tree depth
                    'min_samples_split': 2,   # Minimum samples to split node
                    'min_samples_leaf': 1,    # Minimum samples in leaf
                    'random_state': RANDOM_STATE,
                    'n_jobs': -1              # Use all CPU cores
                }
            },
            'GradientBoosting': {
                'class': GradientBoostingClassifier,
                'params': {
                    'n_estimators': 200,      # Number of boosting stages
                    'max_depth': 5,           # Maximum tree depth
                    'learning_rate': 0.1,     # Shrinkage rate
                    'random_state': RANDOM_STATE
                }
            },
            'LogisticRegression': {
                'class': Pipeline,
                'params': {
                    'steps': [
                        ('scaler', StandardScaler()),
                        ('model', CalibratedClassifierCV(
                            estimator=LogisticRegression(
                                max_iter=3000,
                                random_state=RANDOM_STATE,
                                n_jobs=-1
                            ),
                            method='sigmoid',  
                            cv=5              
                        ))
                    ]
                }
            }
        }
        
        # Add optional models if available
        if 'XGBoost' in MODEL_IMPORTS:
            self.models_config['XGBoost'] = {
                'class': MODEL_IMPORTS['XGBoost'],
                'params': {
                    'n_estimators': 200,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': RANDOM_STATE,
                    'eval_metric': 'mlogloss',  # Evaluation metric
                    'n_jobs': -1
                }
            }
        
        if 'LightGBM' in MODEL_IMPORTS:
            self.models_config['LightGBM'] = {
                'class': MODEL_IMPORTS['LightGBM'],
                'params': {
                    'n_estimators': 200,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': RANDOM_STATE,
                    'verbose': -1,            # Suppress output
                    'n_jobs': -1
                }
            }
        
        if 'CatBoost' in MODEL_IMPORTS:
            self.models_config['CatBoost'] = {
                'class': MODEL_IMPORTS['CatBoost'],
                'params': {
                    'n_estimators': 200,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'random_state': RANDOM_STATE,
                    'verbose': 0,  
                }
            }

    def load_data(self, path: Path = None) -> pd.DataFrame:
        """Load and validate processed training data"""
        if path is None:
            path = PROCESSED / "train.csv"
        
        if not path.exists():
            raise FileNotFoundError(f"Processed data not found: {path}")
        
        # Load CSV file
        df = pd.read_csv(path)
        # Convert date column to datetime if present
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # Validate that target column exists
        if TARGET_COLUMN not in df.columns:
            available_cols = df.columns.tolist()
            logger.error(f"Target column '{TARGET_COLUMN}' not found. Available columns: {available_cols}")
            raise KeyError(f"Target column '{TARGET_COLUMN}' not found in data")
        
        logger.info(f"Loaded data: {df.shape}, target column: '{TARGET_COLUMN}'")
        return df

    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Extract features and target from dataframe"""
        # Define columns to exclude from features (metadata and target)
        exclude_cols = ['date', 'home_team', 'away_team', 'competition', 'season', 
                       'home_goals', 'away_goals', 'team', 'team_away', TARGET_COLUMN]
        
        # Get all columns that are not excluded (these are our features)
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Validate that we have features
        if not feature_cols:
            available_cols = df.columns.tolist()
            raise ValueError(f"No feature columns found. Available columns: {available_cols}")
        
        # Extract feature matrix (X) and target vector (y)
        X = df[feature_cols].fillna(0)  # Fill missing values with 0
        y = df[TARGET_COLUMN].astype(int)  # Ensure target is integer
        
        logger.info(f"Features: {len(feature_cols)}, Samples: {len(X)}, Target: {TARGET_COLUMN}")
        logger.info(f"Feature columns: {feature_cols}")
        
        return X, y, feature_cols

    def temporal_split(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data temporally (chronologically) into train and test sets"""
        if 'date' not in df.columns:
            logger.warning("No date column found, using random split")
            # Fallback to random split if no date column available
            from sklearn.model_selection import train_test_split
            train_df, test_df = train_test_split(df, test_size=test_size, random_state=RANDOM_STATE)
            return train_df, test_df
        
        # Sort by date to ensure chronological order
        df = df.sort_values('date')
        # Calculate split point (e.g., 80% train, 20% test)
        split_idx = int(len(df) * (1 - test_size))
        
        # Split: earlier matches for training, later matches for testing
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")
        logger.info(f"Train date range: {train_df['date'].min()} to {train_df['date'].max()}")
        logger.info(f"Test date range: {test_df['date'].min()} to {test_df['date'].max()}")
        
        return train_df, test_df

    def evaluate_model(self, model, X_train: pd.DataFrame, y_train: pd.Series, 
                      X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model performance on train and test sets"""
        # Get predictions for both sets
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'train_accuracy': accuracy_score(y_train, y_train_pred),        # Training accuracy
            'test_accuracy': accuracy_score(y_test, y_test_pred),           # Test accuracy
            'test_f1_weighted': f1_score(y_test, y_test_pred, average='weighted'),  # Weighted F1
            'test_f1_macro': f1_score(y_test, y_test_pred, average='macro')  # Macro F1
        }
        
        # Calculate log loss if model supports probability predictions
        if hasattr(model, 'predict_proba'):
            y_test_proba = model.predict_proba(X_test)
            metrics['test_log_loss'] = log_loss(y_test, y_test_proba)
        
        logger.info(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
        logger.info(f"Test F1 Weighted: {metrics['test_f1_weighted']:.4f}")
        
        # Print detailed classification report
        logger.info("\nClassification Report:")
        logger.info(classification_report(y_test, y_test_pred, 
                                        target_names=['Away Win', 'Draw', 'Home Win']))
        
        return metrics

    def train_model(self, model_name: str, model_config: Dict, 
                   X_train: pd.DataFrame, y_train: pd.Series,
                   X_test: pd.DataFrame, y_test: pd.Series,
                   feature_names: List[str]) -> Tuple[Any, Dict[str, float]]:
        """Train a single model and log results to MLflow"""
        logger.info(f"Training: {model_name}")
        
        # Start MLflow run for tracking
        with mlflow.start_run(run_name=model_name):
            # Create model instance with configured parameters
            model_class = model_config['class']
            model_params = model_config['params']
            model = model_class(**model_params)
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Evaluate model performance
            metrics = self.evaluate_model(model, X_train, y_train, X_test, y_test)
            
            # Log parameters and metrics to MLflow
            if model_name == 'LogisticRegression':
                mlflow.log_params(model_params['steps'][1][1].get_params())
            else:
                mlflow.log_params(model_params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model, "model")
            
            # Save feature importance if available
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                importance_path = MODELS / f"{model_name}_feature_importance.csv"
                importance_df.to_csv(importance_path, index=False)
                mlflow.log_artifact(str(importance_path))
            
            # Save model to disk
            model_path = MODELS / f"{model_name}_model.pkl"
            joblib.dump(model, model_path)
            
            logger.info(f"Saved: {model_path}")
            return model, metrics

    def train_all_models(self) -> Dict[str, Any]:
        """Train all configured models and compare their performance"""
        # Set MLflow experiment name
        mlflow.set_experiment("football_prediction")
        
        # Load and prepare data
        logger.info("Loading and preparing data...")
        df = self.load_data()
        train_df, test_df = self.temporal_split(df, test_size=0.2)
        
        # Extract features and targets
        X_train, y_train, feature_names = self.prepare_features(train_df)
        X_test, y_test, _ = self.prepare_features(test_df)
        
        results = {}  # Store metrics for each model
        trained_models = {}  # Store trained model objects
        
        # Train each model in configuration
        for model_name, model_config in self.models_config.items():
            try:
                model, metrics = self.train_model(
                    model_name, model_config, X_train, y_train, 
                    X_test, y_test, feature_names
                )
                results[model_name] = metrics
                trained_models[model_name] = model
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        # Validate that at least one model was trained
        if not results:
            raise ValueError("No models were successfully trained")
        
        # Create comparison dataframe and sort by test accuracy
        comparison_df = pd.DataFrame(results).T
        comparison_df = comparison_df.sort_values('test_accuracy', ascending=False)
        
        logger.info("\nModel Comparison:")
        logger.info(comparison_df[['test_accuracy', 'test_f1_weighted']].to_string())
        
        # Save comparison results to CSV
        comparison_path = MODELS / "model_comparison.csv"
        comparison_df.to_csv(comparison_path)
        logger.info(f"Saved comparison: {comparison_path}")
        
        # Identify best model
        best_model_name = comparison_df['test_accuracy'].idxmax()
        best_accuracy = comparison_df.loc[best_model_name, 'test_accuracy']
        
        logger.info(f"Best model: {best_model_name} (Accuracy: {best_accuracy:.4f})")
        
        return {
            'best_model': trained_models[best_model_name],
            'best_model_name': best_model_name,
            'results': results,
            'comparison': comparison_df
        }

def main():
    """Main training function"""
    try:
        trainer = ModelTrainer()
        results = trainer.train_all_models()
        logger.info("Training completed successfully")
        return results
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()