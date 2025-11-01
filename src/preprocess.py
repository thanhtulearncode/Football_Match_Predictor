"""Preprocess football data with feature engineering"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

class EloRating:
    """Elo rating system"""
    def __init__(self, k_factor: float = 20, initial_rating: float = 1500):
        self.k_factor = k_factor
        self.initial_rating = initial_rating
        self.ratings: Dict[str, float] = {}
    
    def get_rating(self, team: str) -> float:
        return self.ratings.get(team, self.initial_rating)
    
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_ratings(self, home_team: str, away_team: str, result: int):
        """Update ratings after match (2=home win, 1=draw, 0=away win)"""
        home_rating = self.get_rating(home_team)
        away_rating = self.get_rating(away_team)
        home_expected = self.expected_score(home_rating, away_rating)
        if result == 2:
            home_actual, away_actual = 1.0, 0.0
        elif result == 0:
            home_actual, away_actual = 0.0, 1.0
        else:
            home_actual, away_actual = 0.5, 0.5
        away_expected = 1 - home_expected
        self.ratings[home_team] = home_rating + self.k_factor * (home_actual - home_expected)
        self.ratings[away_team] = away_rating + self.k_factor * (away_actual - away_expected)
        return home_rating, away_rating

def calculate_form(df: pd.DataFrame, n_matches: int = 5) -> pd.DataFrame:
    """Calculate recent form (points from last N matches)"""
    df = df.sort_values(['date']).copy()
    home_matches = df[['date', 'home_team', 'result']].copy()
    home_matches['team'] = home_matches['home_team']
    home_matches['points'] = home_matches['result'].map({2: 3, 1: 1, 0: 0})
    away_matches = df[['date', 'away_team', 'result']].copy()
    away_matches['team'] = away_matches['away_team']
    away_matches['points'] = away_matches['result'].map({0: 3, 1: 1, 2: 0})
    all_matches = pd.concat([home_matches[['date', 'team', 'points']],
                            away_matches[['date', 'team', 'points']]]).sort_values(['team', 'date'])
    all_matches['form'] = all_matches.groupby('team')['points'].transform(
        lambda x: x.rolling(n_matches, min_periods=1).sum().shift(1))
    return all_matches

def calculate_goal_statistics(df: pd.DataFrame, n_matches: int = 5) -> pd.DataFrame:
    """Calculate rolling goal statistics"""
    df = df.sort_values(['date']).copy()
    home_stats = df[['date', 'home_team', 'home_goals', 'away_goals']].copy()
    home_stats.columns = ['date', 'team', 'goals_scored', 'goals_conceded']
    away_stats = df[['date', 'away_team', 'away_goals', 'home_goals']].copy()
    away_stats.columns = ['date', 'team', 'goals_scored', 'goals_conceded']
    all_stats = pd.concat([home_stats, away_stats]).sort_values(['team', 'date'])
    grouped = all_stats.groupby('team')
    all_stats['avg_goals_scored'] = grouped['goals_scored'].transform(
        lambda x: x.rolling(n_matches, min_periods=1).mean().shift(1))
    all_stats['avg_goals_conceded'] = grouped['goals_conceded'].transform(
        lambda x: x.rolling(n_matches, min_periods=1).mean().shift(1))
    return all_stats

def calculate_head_to_head(df: pd.DataFrame, home_team: str, away_team: str,
                          current_date: str, n_matches: int = 5) -> Dict:
    """Calculate head-to-head statistics"""
    h2h = df[(((df['home_team'] == home_team) & (df['away_team'] == away_team)) |
              ((df['home_team'] == away_team) & (df['away_team'] == home_team))) &
             (df['date'] < current_date)].tail(n_matches)
    if len(h2h) == 0:
        return {'h2h_home_wins': 0, 'h2h_draws': 0, 'h2h_away_wins': 0}
    home_wins = len(h2h[(h2h['home_team'] == home_team) & (h2h['result'] == 2)]) + \
                len(h2h[(h2h['away_team'] == home_team) & (h2h['result'] == 0)])
    away_wins = len(h2h[(h2h['home_team'] == away_team) & (h2h['result'] == 2)]) + \
                len(h2h[(h2h['away_team'] == away_team) & (h2h['result'] == 0)])
    draws = len(h2h[h2h['result'] == 1])
    return {'h2h_home_wins': home_wins, 'h2h_draws': draws, 'h2h_away_wins': away_wins}

def build_features(input_file: str = "matches.csv", output_file: str = "train.csv") -> pd.DataFrame:
    """Build feature set from match data"""
    input_path = RAW / input_file
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    df = pd.read_csv(input_path)
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df = df.sort_values('date').reset_index(drop=True)
    print(f"Loaded {len(df)} matches, {df['home_team'].nunique()} teams")
    elo = EloRating(k_factor=20, initial_rating=1500)
    print("Calculating form...")
    form_df = calculate_form(df, n_matches=5)
    print("Calculating goal statistics...")
    goal_stats = calculate_goal_statistics(df, n_matches=5)
    print("Building features...")
    features_list = []
    for idx, row in df.iterrows():
        if idx % 100 == 0:
            print(f"  {idx}/{len(df)}")
        match_date = row['date']
        home_team = row['home_team']
        away_team = row['away_team']
        home_elo, away_elo = elo.get_rating(home_team), elo.get_rating(away_team)
        elo.update_ratings(home_team, away_team, row['result'])
        home_form = form_df[(form_df['team'] == home_team) & (form_df['date'] == match_date)]['form'].values
        home_form = home_form[0] if len(home_form) > 0 else 0
        away_form = form_df[(form_df['team'] == away_team) & (form_df['date'] == match_date)]['form'].values
        away_form = away_form[0] if len(away_form) > 0 else 0
        home_goals_stats = goal_stats[(goal_stats['team'] == home_team) & (goal_stats['date'] == match_date)]
        home_avg_scored = home_goals_stats['avg_goals_scored'].values[0] if len(home_goals_stats) > 0 else 1.0
        home_avg_conceded = home_goals_stats['avg_goals_conceded'].values[0] if len(home_goals_stats) > 0 else 1.0
        away_goals_stats = goal_stats[(goal_stats['team'] == away_team) & (goal_stats['date'] == match_date)]
        away_avg_scored = away_goals_stats['avg_goals_scored'].values[0] if len(away_goals_stats) > 0 else 1.0
        away_avg_conceded = away_goals_stats['avg_goals_conceded'].values[0] if len(away_goals_stats) > 0 else 1.0
        h2h = calculate_head_to_head(df, home_team, away_team, match_date, n_matches=5)
        home_last_match = df[((df['home_team'] == home_team) | (df['away_team'] == home_team)) &
                             (df['date'] < match_date)]['date'].max()
        days_since_home = (match_date - home_last_match).days if pd.notna(home_last_match) else 14
        away_last_match = df[((df['home_team'] == away_team) | (df['away_team'] == away_team)) &
                             (df['date'] < match_date)]['date'].max()
        days_since_away = (match_date - away_last_match).days if pd.notna(away_last_match) else 14
        features = {
            'date': match_date, 'home_team': home_team, 'away_team': away_team,
            'home_elo': home_elo, 'away_elo': away_elo, 'elo_diff': home_elo - away_elo,
            'home_form': home_form, 'away_form': away_form, 'form_diff': home_form - away_form,
            'home_avg_scored': home_avg_scored, 'home_avg_conceded': home_avg_conceded,
            'away_avg_scored': away_avg_scored, 'away_avg_conceded': away_avg_conceded,
            'attack_strength_diff': home_avg_scored - away_avg_conceded,
            'h2h_home_wins': h2h['h2h_home_wins'], 'h2h_draws': h2h['h2h_draws'],
            'h2h_away_wins': h2h['h2h_away_wins'],
            'home_rest_days': days_since_home, 'away_rest_days': days_since_away,
            'rest_advantage': days_since_home - days_since_away, 'target': row['result']
        }
        features_list.append(features)
    features_df = pd.DataFrame(features_list)
    features_df = features_df.fillna(0)
    output_path = PROCESSED / output_file
    features_df.to_csv(output_path, index=False)
    print(f"Saved {features_df.shape[0]} matches, {features_df.shape[1]} features")
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    correlations = features_df[numeric_cols].corr()['target'].sort_values(ascending=False)
    print(f"Top correlations: {correlations.head(5).to_dict()}")
    return features_df

if __name__ == "__main__":
    try:
        df = build_features()
        print("Preprocessing complete")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Run data_fetcher.py first")
