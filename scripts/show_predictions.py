"""CLI script to display predictions for upcoming matches."""
import sys
import logging
from pathlib import Path
import pandas as pd
import joblib

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.settings import settings
from src.pipeline.processing import preprocess_time_features, apply_encoders

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def _get_class_indices(classes):
    """Extract class indices for Win/Draw/Loss."""
    class_list = list(classes)
    try:
        return (
            class_list.index("W"),
            class_list.index("D"),
            class_list.index("L")
        )
    except ValueError:
        return (2, 1, 0)


def _format_prediction_string(pred):
    """Convert numeric prediction to human-readable string."""
    prediction_map = {2.0: "Home Win", 1.0: "Draw", 0.0: "Away Win"}
    return prediction_map.get(float(pred), "Draw")


def _load_model_and_data():
    """Load the trained model and upcoming matches data."""
    model_path = settings.MODEL_PATH
    upcoming_path = settings.UPCOMING_FINAL_PATH
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Please train the model first."
        )
    
    if not upcoming_path.exists():
        raise FileNotFoundError(
            f"Upcoming matches data not found at {upcoming_path}. "
            "Please run the pipeline first."
        )
    
    logger.info("Loading model and upcoming matches...")
    model = joblib.load(model_path)
    upcoming = pd.read_csv(upcoming_path)
    
    if upcoming.empty:
        raise ValueError("No upcoming matches found in the processed data.")
    
    return model, upcoming


def _validate_features(upcoming):
    """Validate that all required predictor columns exist."""
    missing_cols = [col for col in settings.PREDICTORS if col not in upcoming.columns]
    if missing_cols:
        raise ValueError(
            f"Missing required columns in upcoming data: {missing_cols}. "
            "Please re-run the pipeline."
        )


def _prepare_features(upcoming):
    """Prepare features for prediction by filtering and applying encoders."""
    # Filter for home matches only
    if "venue" in upcoming.columns:
        upcoming = upcoming[upcoming["venue"].isin(settings.HOME_VENUES)].copy()
    
    # Load and apply encoders if available
    encoders_path = settings.ENCODERS_PATH
    if encoders_path.exists():
        encoders = joblib.load(encoders_path)
        upcoming = preprocess_time_features(upcoming)
        upcoming = apply_encoders(upcoming, encoders)
    else:
        logger.warning("Encoders not found. Features may not be properly encoded.")
    
    return upcoming


def _generate_predictions(model, upcoming):
    """Generate predictions for upcoming matches."""
    X = upcoming[settings.PREDICTORS].fillna(0)
    
    logger.info(f"Generating predictions for {len(upcoming)} matches...")
    preds = model.predict(X)
    probs = model.predict_proba(X)
    
    return preds, probs


def _format_results(upcoming, preds, probs, model):
    """Format prediction results for display."""
    win_idx, draw_idx, loss_idx = _get_class_indices(model.classes_)
    
    results = upcoming[["date", "time", "team", "opponent"]].copy()
    results["prediction"] = preds
    results["home_win_prob"] = probs[:, win_idx]
    results["draw_prob"] = probs[:, draw_idx]
    results["away_win_prob"] = probs[:, loss_idx]
    results["confidence"] = probs.max(axis=1)
    results["prediction_str"] = results["prediction"].apply(_format_prediction_string)
    
    return results.sort_values(["date", "time"])


def _display_results(results, limit=20):
    """Display formatted prediction results."""
    print("\n" + "=" * 90)
    print("⚽ UPCOMING MATCH PREDICTIONS ⚽".center(90))
    print("=" * 90)
    print(
        f"{'Date':<12} {'Time':<6} {'Home Team':<25} vs {'Away Team':<25} "
        f"{'Prediction':<12} {'Conf':<6}"
    )
    print("-" * 90)
    
    for _, row in results.head(limit).iterrows():
        home = str(row['team'])[:24]
        away = str(row['opponent'])[:24]
        pred = str(row['prediction_str'])[:11]
        conf = f"{row['confidence']:.1%}"
        date_str = str(row['date'])[:10] if len(str(row['date'])) > 10 else str(row['date'])
        time_str = str(row.get('time', ''))[:5]
        
        print(
            f"{date_str:<12} {time_str:<6} {home:<25} vs {away:<25} "
            f"{pred:<12} {conf:<6}"
        )
    
    print("=" * 90)
    if len(results) > limit:
        print(f"\nShowing {limit} of {len(results)} upcoming matches.")
    print(
        "\nNote: Probabilities indicate the likelihood of Home Win, Draw, or Away Win."
    )


def show_predictions(limit=20):
    """Display predictions for upcoming matches."""
    try:
        # Load data
        model, upcoming = _load_model_and_data()
        
        # Validate and prepare
        _validate_features(upcoming)
        prepared = _prepare_features(upcoming)
        
        if prepared.empty:
            logger.warning("No home matches found after filtering.")
            return
        
        # Generate predictions
        preds, probs = _generate_predictions(model, prepared)
        
        # Format and display
        results = _format_results(prepared, preds, probs, model)
        _display_results(results, limit=limit)
        
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    show_predictions()