"""Preprocessing with vectorized operations"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import logging
from config import RAW_DIR as RAW, PROCESSED_DIR as PROCESSED, ELO_K_FACTOR, FORM_MATCHES_N, GOAL_STAT_MATCHES

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EloRating:
    """Elo rating system for calculating team strength ratings"""
    def __init__(self, k_factor: float = ELO_K_FACTOR, initial_rating: float = 1500):
        # K-factor controls how much ratings change per match
        self.k_factor = k_factor
        # Starting rating for new teams
        self.initial_rating = initial_rating
        # Dictionary to store current ratings for each team
        self.ratings: Dict[str, float] = {}
    
    def get_rating(self, team: str) -> float:
        """Get current rating for a team, or return initial rating if team is new"""
        return self.ratings.get(team, self.initial_rating)
    
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score for team A against team B using Elo formula"""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def batch_update(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Update Elo ratings for all matches and add ratings to dataframe"""
        ratings_before = []
        
        # Process each match chronologically
        for _, row in matches.iterrows():
            home_team, away_team, result = row['home_team'], row['away_team'], row['result']
            home_rating = self.get_rating(home_team)
            away_rating = self.get_rating(away_team)
            
            # Store ratings before update (for feature engineering)
            ratings_before.append((home_rating, away_rating))
            
            # Calculate expected scores
            home_expected = self.expected_score(home_rating, away_rating)
            away_expected = 1 - home_expected
            
            # Determine actual scores based on match result
            if result == 2:  # Home win
                home_actual, away_actual = 1.0, 0.0
            elif result == 0:  # Away win
                home_actual, away_actual = 0.0, 1.0
            else:  # Draw
                home_actual, away_actual = 0.5, 0.5
            
            # Update ratings: new_rating = old_rating + K * (actual - expected)
            self.ratings[home_team] = home_rating + self.k_factor * (home_actual - home_expected)
            self.ratings[away_team] = away_rating + self.k_factor * (away_actual - away_expected)
        
        # Add Elo ratings as features to matches dataframe
        ratings_df = pd.DataFrame(ratings_before, columns=['home_elo', 'away_elo'])
        return pd.concat([matches.reset_index(drop=True), ratings_df], axis=1)

def calculate_rolling_stats(df: pd.DataFrame, n_matches: int) -> pd.DataFrame:
    """Calculate rolling form points and goal statistics for each team"""
    # Create home team perspective: each match as if team played at home
    home_matches = df[['date', 'home_team', 'result', 'home_goals', 'away_goals']].copy()
    home_matches['team'] = home_matches['home_team']
    # Map result to points: 3 for win, 1 for draw, 0 for loss
    home_matches['points'] = home_matches['result'].map({2: 3, 1: 1, 0: 0})
    home_matches['goals_scored'] = home_matches['home_goals']
    home_matches['goals_conceded'] = home_matches['away_goals']
    
    # Create away team perspective: each match as if team played away
    away_matches = df[['date', 'away_team', 'result', 'away_goals', 'home_goals']].copy()
    away_matches['team'] = away_matches['away_team']
    # Reverse point mapping for away team (result 0=away win, 2=home win)
    away_matches['points'] = away_matches['result'].map({0: 3, 1: 1, 2: 0})
    away_matches['goals_scored'] = away_matches['away_goals']
    away_matches['goals_conceded'] = away_matches['home_goals']
    
    # Combine home and away perspectives into single view
    all_matches = pd.concat([
        home_matches[['date', 'team', 'points', 'goals_scored', 'goals_conceded']],
        away_matches[['date', 'team', 'points', 'goals_scored', 'goals_conceded']]
    ]).sort_values(['team', 'date'])
    
    # Calculate rolling statistics per team
    grouped = all_matches.groupby('team')
    # Form: sum of points from last n matches (shifted to exclude current match)
    all_matches['form'] = grouped['points'].transform(
        lambda x: x.rolling(n_matches, min_periods=1).sum().shift(1))
    # Average goals scored in last n matches
    all_matches['avg_goals_scored'] = grouped['goals_scored'].transform(
        lambda x: x.rolling(n_matches, min_periods=1).mean().shift(1))
    # Average goals conceded in last n matches
    all_matches['avg_goals_conceded'] = grouped['goals_conceded'].transform(
        lambda x: x.rolling(n_matches, min_periods=1).mean().shift(1))
    
    return all_matches

def build_features(input_file: str = "matches.csv", output_file: str = "train.csv") -> pd.DataFrame:
    """Build feature set from raw match data"""
    input_path = RAW / input_file
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Load and prepare raw data
    logger.info("Loading data...")
    df = pd.read_csv(input_path)
    df['date'] = pd.to_datetime(df['date'])
    # Sort by date to ensure chronological processing
    df = df.sort_values('date').reset_index(drop=True)
    
    logger.info(f"Processing {len(df)} matches, {df['home_team'].nunique()} teams")
    
    # Step 1: Calculate Elo ratings for all teams
    logger.info("Calculating Elo ratings...")
    elo = EloRating(initial_rating=1500)
    df_with_elo = elo.batch_update(df)
    
    # Step 2: Calculate form and goal statistics
    logger.info("Calculating form and goal statistics...")
    stats_df = calculate_rolling_stats(df, FORM_MATCHES_N)
    
    # Step 3: Merge all features
    logger.info("Building features...")
    
    # Extract home team features (form, goals scored/conceded when playing at home)
    home_features = stats_df[stats_df['team'].isin(df['home_team'])].rename(
        columns={'form': 'home_form', 'avg_goals_scored': 'home_avg_scored', 
                'avg_goals_conceded': 'home_avg_conceded'})
    home_features = home_features[['date', 'team', 'home_form', 'home_avg_scored', 'home_avg_conceded']]
    
    # Extract away team features (form, goals scored/conceded when playing away)
    away_features = stats_df[stats_df['team'].isin(df['away_team'])].rename(
        columns={'form': 'away_form', 'avg_goals_scored': 'away_avg_scored',
                'avg_goals_conceded': 'away_avg_conceded'})
    away_features = away_features[['date', 'team', 'away_form', 'away_avg_scored', 'away_avg_conceded']]
    
    # Merge home and away features with main dataframe
    features_df = df_with_elo.merge(
        home_features, 
        left_on=['date', 'home_team'], 
        right_on=['date', 'team'], 
        how='left'
    ).merge(
        away_features,
        left_on=['date', 'away_team'], 
        right_on=['date', 'team'],
        how='left',
        suffixes=('', '_away')
    )
    
    # Step 4: Calculate derived features (differences between teams)
    features_df['elo_diff'] = features_df['home_elo'] - features_df['away_elo']
    features_df['form_diff'] = features_df['home_form'] - features_df['away_form']
    features_df['attack_strength_diff'] = features_df['home_avg_scored'] - features_df['away_avg_conceded']
    
    # Handle missing values (for teams with no historical data)
    features_df = features_df.fillna(0)
    
    # Save processed data to CSV
    output_path = PROCESSED / output_file
    features_df.to_csv(output_path, index=False)
    
    logger.info(f"Saved {len(features_df)} matches with {len(features_df.columns)} features")
    
    return features_df

if __name__ == "__main__":
    try:
        df = build_features()
        logger.info("Preprocessing complete")
    except FileNotFoundError as e:
        logger.error(f"Error: {e}")
        logger.info("Run data_fetcher.py first")