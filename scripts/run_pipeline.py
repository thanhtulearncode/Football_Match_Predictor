import sys
import logging
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.settings import settings
from src.pipeline.sources.fbref import FbrefScraper
from src.pipeline.processing import clean_data, preprocess_time_features
from src.pipeline.features import add_elo_features, add_rolling_features, add_opponent_stats
from src.pipeline.train import ModelTrainer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def _setup_paths():
    return {
        'raw_matches': settings.RAW_MATCHES_PATH,
        'upcoming_raw': settings.UPCOMING_RAW_PATH,
        'matches_cleaned': settings.MATCHES_CLEANED_PATH,
        'matches_final': settings.MATCHES_FINAL_PATH,
        'upcoming_final': settings.UPCOMING_FINAL_PATH,
    }


def _scrape_data():
    logger.info(f"Starting Data Scraping...")
    scraper = FbrefScraper(data_dir=settings.RAW_DIR)
    scraper.scrape_premier_league(
        start_year=settings.SCRAPING_START_YEAR,
        end_year=settings.SCRAPING_END_YEAR
    )
    logger.info("Data scraping completed.")


def _load_data(paths):
    logger.info(f"Loading Data...")
    if not paths['raw_matches'].exists():
        raise FileNotFoundError(
            f"No historical matches found at {paths['raw_matches']}. "
            "Please run data scraping first."
        )
    
    matches = pd.read_csv(paths['raw_matches'])
    matches["is_future"] = False
    # Load upcoming matches if available
    if paths['upcoming_raw'].exists():
        upcoming = pd.read_csv(paths['upcoming_raw'])
        upcoming["is_future"] = True
        full_df = pd.concat([matches, upcoming], ignore_index=True)
        logger.info(
            f"Loaded {len(matches)} historical and {len(upcoming)} upcoming matches."
        )
    else:
        full_df = matches
        logger.info(f"Loaded {len(matches)} historical matches.")
    
    return full_df


def _clean_data(df, output_path):
    """Clean and validate the raw match data."""
    logger.info(f"Cleaning Data...")
    cleaned_df = clean_data(df)
    cleaned_df.to_csv(output_path, index=False)
    logger.info(f"Cleaned data saved to {output_path}")
    return cleaned_df


def _engineer_features(df):
    """Add Elo ratings, rolling stats, and opponent statistics."""
    logger.info(f"Engineering Features...")
    
    logger.info("   - Calculating Elo Ratings...")
    df = add_elo_features(df)
    
    logger.info("   - Calculating Rolling Stats...")
    df, _ = add_rolling_features(df)
    
    logger.info("   - Merging Opponent Stats...")
    df, _ = add_opponent_stats(df)
    
    return df


def _split_and_save_data(df, paths):
    """Split data into training and upcoming sets, then save."""
    matches_final = df[~df["is_future"]].drop(columns=["is_future"])
    upcoming_final = df[df["is_future"]].drop(columns=["is_future"])
    
    matches_final.to_csv(paths['matches_final'], index=False)
    logger.info(f"Saved {len(matches_final)} training matches to {paths['matches_final']}")
    
    if not upcoming_final.empty:
        upcoming_final.to_csv(paths['upcoming_final'], index=False)
        logger.info(
            f"Saved {len(upcoming_final)} upcoming matches to {paths['upcoming_final']}"
        )
    else:
        logger.warning("No upcoming matches to save.")


def _train_model(data_path):
    """Train the prediction model."""
    logger.info(f"Training Model...")
    trainer = ModelTrainer(data_path=data_path)
    trainer.train()
    logger.info("Model training completed.")


def run_pipeline():
    """Execute the complete pipeline: scrape -> process -> train."""
    logger.info("=" * 60)
    logger.info("Starting Football Predictor Pipeline")
    logger.info("=" * 60)
    
    try:
        paths = _setup_paths()
        _scrape_data()
        full_df = _load_data(paths)
        cleaned_df = _clean_data(full_df, paths['matches_cleaned'])
        feature_df = _engineer_features(cleaned_df)
        _split_and_save_data(feature_df, paths)
        _train_model(paths['matches_final'])
        
        logger.info("=" * 60)
        logger.info("Pipeline Complete!")
        logger.info("=" * 60)
        
    except FileNotFoundError as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed with unexpected error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    run_pipeline()