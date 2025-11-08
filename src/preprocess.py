"""Preprocessing with vectorized operations"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import logging
import json
from datetime import datetime, timedelta
from config import (
    RAW_DIR as RAW, PROCESSED_DIR as PROCESSED, ELO_K_FACTOR, 
    FORM_MATCHES_N, GOAL_STAT_MATCHES, TEAM_STATS_FILE, TARGET_COLUMN,
    TEAM_NAME_MAPPINGS
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EloRating:
    """Elo rating system for calculating team strength ratings"""
    def __init__(self, k_factor: float = ELO_K_FACTOR, initial_rating: float = 1500):
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.ratings: Dict[str, float] = {}
        self.matches_played: Dict[str, int] = {}
    
    def get_rating(self, team: str) -> float:
        return self.ratings.get(team, self.initial_rating)
    
    def get_matches_played(self, team: str) -> int:
        return self.matches_played.get(team, 0)
    
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_ratings(self, home_team: str, away_team: str, result: int, home_advantage: float = 100):
        """Update ratings after match with home advantage"""
        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)
        
        # Apply home advantage
        home_rating_adj = home_rating + home_advantage
        
        home_expected = self.expected_score(home_rating_adj, away_rating)
        
        if result == 2:  # Home win
            home_actual, away_actual = 1.0, 0.0
        elif result == 0:  # Away win
            home_actual, away_actual = 0.0, 1.0
        else:  # Draw
            home_actual, away_actual = 0.5, 0.5
            
        away_expected = 1 - home_expected
        
        # Update ratings
        self.ratings[home_team] = home_rating + self.k_factor * (home_actual - home_expected)
        self.ratings[away_team] = away_rating + self.k_factor * (away_actual - away_expected)
        
        # Update matches played
        self.matches_played[home_team] = self.matches_played.get(home_team, 0) + 1
        self.matches_played[away_team] = self.matches_played.get(away_team, 0) + 1
        
        return home_rating, away_rating

def normalize_team_name(team_name: str) -> str:
    """Normalize team names using mappings"""
    for canonical_name, variants in TEAM_NAME_MAPPINGS.items():
        if team_name in variants or team_name == canonical_name:
            return canonical_name
    return team_name

def calculate_team_statistics(df: pd.DataFrame) -> Dict:
    """Calculate statistics for all teams"""
    team_stats = {}
    
    # Get unique teams
    all_teams = set(df['home_team'].unique()) | set(df['away_team'].unique())
    
    for team in all_teams:
        # Team's matches (both home and away)
        home_matches = df[df['home_team'] == team].copy()
        away_matches = df[df['away_team'] == team].copy()
        
        # Combine all matches
        home_matches_processed = home_matches[['date', 'home_team', 'away_team', 'home_goals', 'away_goals', 'result']].rename(
            columns={'home_team': 'team', 'away_team': 'opponent', 'home_goals': 'goals_for', 'away_goals': 'goals_against'}
        )
        home_matches_processed['is_home'] = True
        
        away_matches_processed = away_matches[['date', 'home_team', 'away_team', 'home_goals', 'away_goals', 'result']].rename(
            columns={'away_team': 'team', 'home_team': 'opponent', 'away_goals': 'goals_for', 'home_goals': 'goals_against'}
        )
        away_matches_processed['is_home'] = False
        
        all_team_matches = pd.concat([home_matches_processed, away_matches_processed]).sort_values('date')
        
        # Calculate form (points from last 5 matches)
        recent_matches = all_team_matches.tail(5)
        form = 0
        for _, match in recent_matches.iterrows():
            if match['is_home']:
                if match['result'] == 2: form += 3  # Home win
                elif match['result'] == 1: form += 1  # Draw
            else:
                if match['result'] == 0: form += 3  # Away win
                elif match['result'] == 1: form += 1  # Draw
        
        # Goal statistics (last 5 matches)
        if len(recent_matches) > 0:
            avg_scored = recent_matches['goals_for'].mean()
            avg_conceded = recent_matches['goals_against'].mean()
            goal_difference = avg_scored - avg_conceded
        else:
            avg_scored = avg_conceded = goal_difference = 1.5
        
        # Overall statistics
        total_matches = len(all_team_matches)
        if total_matches > 0:
            win_rate = len(all_team_matches[
                (all_team_matches['is_home'] & (all_team_matches['result'] == 2)) |
                (~all_team_matches['is_home'] & (all_team_matches['result'] == 0))
            ]) / total_matches
        else:
            win_rate = 0.33
        
        team_stats[team] = {
            'form': form,
            'avg_scored': float(avg_scored),
            'avg_conceded': float(avg_conceded),
            'goal_difference': float(goal_difference),
            'win_rate': float(win_rate),
            'total_matches': total_matches,
            'recent_matches_count': len(recent_matches)
        }
    
    return team_stats

def calculate_head_to_head(df: pd.DataFrame, home_team: str, away_team: str, current_date: str, n_matches: int = 10) -> Dict:
    """Calculate head-to-head statistics"""
    h2h_matches = df[
        (((df['home_team'] == home_team) & (df['away_team'] == away_team)) |
         ((df['home_team'] == away_team) & (df['away_team'] == home_team))) &
        (df['date'] < current_date)
    ].tail(n_matches)
    
    home_wins = 0
    draws = 0
    away_wins = 0
    total_goals_home = 0
    total_goals_away = 0
    
    for _, match in h2h_matches.iterrows():
        if match['home_team'] == home_team:
            # Home team is the reference
            if match['result'] == 2: 
                home_wins += 1
            elif match['result'] == 1: 
                draws += 1
            else: 
                away_wins += 1
            total_goals_home += match['home_goals']
            total_goals_away += match['away_goals']
        else:
            # Away team is the reference (reverse the result)
            if match['result'] == 0: 
                home_wins += 1  # Original away win becomes home win
            elif match['result'] == 1: 
                draws += 1
            else: 
                away_wins += 1  # Original home win becomes away win
            total_goals_away += match['home_goals']  # Reverse goal counting
            total_goals_home += match['away_goals']
    
    total_matches = len(h2h_matches)
    if total_matches > 0:
        home_win_rate = home_wins / total_matches
        avg_goals_home = total_goals_home / total_matches
        avg_goals_away = total_goals_away / total_matches
    else:
        home_win_rate = avg_goals_home = avg_goals_away = 0
    
    return {
        'h2h_home_wins': home_wins,
        'h2h_draws': draws,
        'h2h_away_wins': away_wins,
        'h2h_home_win_rate': home_win_rate,
        'h2h_avg_goals_home': avg_goals_home,
        'h2h_avg_goals_away': avg_goals_away,
        'h2h_total_matches': total_matches
    }

def calculate_rest_days(df: pd.DataFrame, team: str, current_date: str) -> int:
    """Calculate rest days for a team before a match"""
    team_matches = df[
        ((df['home_team'] == team) | (df['away_team'] == team)) &
        (df['date'] < current_date)
    ].tail(1)
    
    if len(team_matches) == 0:
        return 7  # Default rest if no previous matches
    
    last_match_date = team_matches.iloc[0]['date']
    days_rest = (current_date - last_match_date).days
    return max(1, min(days_rest, 30))  # Clamp between 1 and 30 days

def build_features_and_stats(input_file: str = "matches.csv", output_file: str = "train.csv") -> pd.DataFrame:
    """Build feature set from match data"""
    input_path = RAW / input_file
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    logger.info("Loading raw data...")
    df = pd.read_csv(input_path)
    
    # Ensure required columns exist
    required_cols = ['date', 'home_team', 'away_team', 'home_goals', 'away_goals', 'result']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in raw data: {missing_cols}")
    
    # Normalize team names
    logger.info("Normalizing team names...")
    df['home_team'] = df['home_team'].apply(normalize_team_name)
    df['away_team'] = df['away_team'].apply(normalize_team_name)
    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    logger.info(f"Processing {len(df)} matches, {df['home_team'].nunique()} teams")
    
    # Initialize Elo system
    elo = EloRating(initial_rating=1500)
    
    # Calculate team statistics
    logger.info("Calculating team statistics...")
    team_stats = calculate_team_statistics(df)
    
    # Build features for training
    logger.info("Building features...")
    features_list = []
    
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            logger.info(f"Processing match {idx}/{len(df)}")
        
        match_date = row['date']
        home_team = row['home_team']
        away_team = row['away_team']
        result = row['result']
        
        # Get Elo ratings before update
        home_elo, away_elo = elo.get_rating(home_team), elo.get_rating(away_team)
        
        # Update Elo ratings
        elo.update_ratings(home_team, away_team, result)
        
        # Get team statistics
        home_stat = team_stats.get(home_team, {'form': 0, 'avg_scored': 1.5, 'avg_conceded': 1.5, 'win_rate': 0.33})
        away_stat = team_stats.get(away_team, {'form': 0, 'avg_scored': 1.5, 'avg_conceded': 1.5, 'win_rate': 0.33})
        
        # Calculate head-to-head
        h2h = calculate_head_to_head(df, home_team, away_team, match_date)
        
        # Calculate rest days
        home_rest = calculate_rest_days(df, home_team, match_date)
        away_rest = calculate_rest_days(df, away_team, match_date)
        
        # Build comprehensive features
        features = {
            'date': match_date,
            'home_team': home_team,
            'away_team': away_team,
            'home_goals': row['home_goals'],
            'away_goals': row['away_goals'],
            # Elo features
            'home_elo': home_elo,
            'away_elo': away_elo,
            'elo_diff': home_elo - away_elo,
            # Form features
            'home_form': home_stat['form'],
            'away_form': away_stat['form'],
            'form_diff': home_stat['form'] - away_stat['form'],
            # Goal statistics
            'home_avg_scored': home_stat['avg_scored'],
            'home_avg_conceded': home_stat['avg_conceded'],
            'away_avg_scored': away_stat['avg_scored'],
            'away_avg_conceded': away_stat['avg_conceded'],
            'attack_strength_diff': home_stat['avg_scored'] - away_stat['avg_conceded'],
            'defense_strength_diff': away_stat['avg_scored'] - home_stat['avg_conceded'],
            # Head-to-head features
            'h2h_home_wins': h2h['h2h_home_wins'],
            'h2h_draws': h2h['h2h_draws'],
            'h2h_away_wins': h2h['h2h_away_wins'],
            'h2h_home_win_rate': h2h['h2h_home_win_rate'],
            'h2h_avg_goals_home': h2h['h2h_avg_goals_home'],
            'h2h_avg_goals_away': h2h['h2h_avg_goals_away'],
            # Rest and recovery
            'home_rest_days': home_rest,
            'away_rest_days': away_rest,
            'rest_advantage': home_rest - away_rest,
            # Additional features
            'home_win_rate': home_stat['win_rate'],
            'away_win_rate': away_stat['win_rate'],
            'win_rate_diff': home_stat['win_rate'] - away_stat['win_rate'],
            # Target
            TARGET_COLUMN: result
        }
        
        features_list.append(features)
    
    features_df = pd.DataFrame(features_list)
    
    # Fill NaN values
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    features_df[numeric_cols] = features_df[numeric_cols].fillna(0)
    
    # Save training data
    output_path = PROCESSED / output_file
    features_df.to_csv(output_path, index=False)
    
    # Save team statistics with current Elo ratings
    for team in elo.ratings:
        if team in team_stats:
            team_stats[team]['elo'] = elo.ratings[team]
            team_stats[team]['matches_played'] = elo.get_matches_played(team)
        else:
            team_stats[team] = {
                'elo': elo.ratings[team],
                'matches_played': elo.get_matches_played(team),
                'form': 0, 
                'avg_scored': 1.5, 
                'avg_conceded': 1.5,
                'win_rate': 0.33
            }
    
    with open(TEAM_STATS_FILE, 'w') as f:
        json.dump(team_stats, f, indent=2, default=str)
    
    logger.info(f"Saved {len(features_df)} matches with {len(features_df.columns)} features")
    logger.info(f"Saved statistics for {len(team_stats)} teams")
    logger.info(f"Target distribution: {features_df[TARGET_COLUMN].value_counts().sort_index()}")
    
    # Print feature correlations
    numeric_features = features_df.select_dtypes(include=[np.number])
    if TARGET_COLUMN in numeric_features.columns:
        correlations = numeric_features.corr()[TARGET_COLUMN].sort_values(ascending=False)
        logger.info("Top feature correlations with target:")
        for feature, corr in correlations.head(10).items():
            logger.info(f"  {feature}: {corr:.3f}")
    
    return features_df

if __name__ == "__main__":
    try:
        df = build_features_and_stats()
        logger.info("Preprocessing complete!")
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        raise