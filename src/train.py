"""Train football prediction models"""
import pandas as pd
from pathlib import Path
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
import joblib

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS = Path(__file__).resolve().parent.parent / "models"
MODELS.mkdir(parents=True, exist_ok=True)

def load_data(path=None):
    """Load processed data"""
    if path is None:
        path = PROCESSED / "train.csv"
    df = pd.read_csv(path)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    return df

def prepare_X_y(df: pd.DataFrame):
    """Prepare features and target"""
    exclude_cols = ['date', 'home_team', 'away_team', 'target']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].fillna(0)
    y = df['target'].astype(int)
    print(f"Features: {len(feature_cols)}, Shape: {X.shape}")
    return X, y, feature_cols

def temporal_train_test_split(df: pd.DataFrame, test_size: float = 0.2):
    """Temporal split to avoid data leakage"""
    df = df.sort_values('date')
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")
    return train_df, test_df

def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluate model"""
    y_test_pred = model.predict(X_test)
    metrics = {
        'train_accuracy': accuracy_score(y_train, model.predict(X_train)),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'test_f1_weighted': f1_score(y_test, y_test_pred, average='weighted')
    }
    print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    print(classification_report(y_test, y_test_pred, target_names=['Away', 'Draw', 'Home']))
    return metrics

def train_single_model(model_name: str, model, X_train, y_train, X_test, y_test, feature_names):
    """Train and evaluate single model"""
    print(f"\nTraining: {model_name}")
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_type", model_name)
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_test", len(X_test))
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_train, y_train, X_test, y_test)
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, float(value))
        if hasattr(model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            importance_path = MODELS / f"{model_name}_feature_importance.csv"
            importance.to_csv(importance_path, index=False)
            mlflow.log_artifact(str(importance_path))
        mlflow.sklearn.log_model(model, "model")
        model_path = MODELS / f"{model_name}_model.pkl"
        joblib.dump(model, model_path)
        print(f"Saved: {model_path}")
        return model, metrics

def train_all_models():
    """Train and compare all models"""
    mlflow.set_experiment("football_prediction")
    print("Loading data...")
    df = load_data()
    train_df, test_df = temporal_train_test_split(df, test_size=0.2)
    X_train, y_train, feature_names = prepare_X_y(train_df)
    X_test, y_test, _ = prepare_X_y(test_df)
    models = {}
    if XGB_AVAILABLE:
        models['XGBoost'] = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                          random_state=42, eval_metric='mlogloss')
    if LGBM_AVAILABLE:
        models['LightGBM'] = LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                           random_state=42, verbose=-1)
    if CATBOOST_AVAILABLE:
        models['CatBoost'] = CatBoostClassifier(iterations=200, depth=6, learning_rate=0.1,
                                               random_state=42, verbose=False)
    models['RandomForest'] = RandomForestClassifier(n_estimators=200, max_depth=10,
                                                    random_state=42)
    models['GradientBoosting'] = GradientBoostingClassifier(n_estimators=200, max_depth=5,
                                                            learning_rate=0.1, random_state=42)
    models['LogisticRegression'] = LogisticRegression(max_iter=2000, random_state=42)
    results = {}
    trained_models = {}
    for model_name, model in models.items():
        try:
            trained_model, metrics = train_single_model(
                model_name, model, X_train, y_train, X_test, y_test, feature_names)
            results[model_name] = metrics
            trained_models[model_name] = trained_model
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            continue
    comparison = pd.DataFrame(results).T
    comparison = comparison.sort_values('test_accuracy', ascending=False)
    print("\nModel Comparison:")
    print(comparison[['test_accuracy', 'test_f1_weighted']].to_string())
    comparison_path = MODELS / "model_comparison.csv"
    comparison.to_csv(comparison_path)
    print(f"Saved: {comparison_path}")
    best_model_name = comparison['test_accuracy'].idxmax()
    print(f"\nBest: {best_model_name} ({results[best_model_name]['test_accuracy']:.4f})")
    return trained_models[best_model_name], best_model_name, results

if __name__ == "__main__":
    try:
        best_model, best_model_name, results = train_all_models()
        print(f"\nTraining complete. Best: {best_model_name}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run preprocess.py first")
    except Exception as e:
        print(f"Error: {e}")
