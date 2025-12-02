import pandas as pd
import logging
from typing import Tuple
from config.settings import settings

logger = logging.getLogger(__name__)


def add_elo_features(df: pd.DataFrame, k_factor: int = None, initial_rating: float = None) -> pd.DataFrame:
    """Calculate Elo ratings"""
    k_factor = k_factor or settings.ELO_K_FACTOR
    initial_rating = initial_rating or settings.ELO_INITIAL_RATING
    df = df.sort_values("date").reset_index(drop=True)
    all_teams = pd.concat([df["team"], df["opponent"]]).unique()
    elo_dict = {team: initial_rating for team in all_teams}
    team_elos = []
    opp_elos = []
    TARGET_TO_SCORE = settings.ELO_TARGET_TO_SCORE
    for _, row in df.iterrows():
        team = row["team"]
        opp = row["opponent"]
        
        team_elo = elo_dict[team]
        opp_elo = elo_dict[opp]
        
        team_elos.append(team_elo)
        opp_elos.append(opp_elo)
        
        target = row.get("target")
        if pd.notna(target):
            target_int = int(target)
            expected = 1 / (1 + 10 ** ((opp_elo - team_elo) / 400))
            actual = TARGET_TO_SCORE.get(target_int, 0.0)
            update = k_factor * (actual - expected)
            elo_dict[team] += update
            elo_dict[opp] -= update
            
    df["elo"] = team_elos
    df["elo_opp"] = opp_elos
    return df

def add_rolling_features(df: pd.DataFrame, window: int = None, cols: list = None) -> Tuple[pd.DataFrame, list]:
    """Adds rolling averages (form)"""
    window = window or settings.ROLLING_WINDOW
    cols = cols or settings.ROLLING_STATS_COLS
    existing_cols = [c for c in cols if c in df.columns]
    
    if not existing_cols:
        return df, []
        
    df = df.sort_values("date")
    
    rolling_stats = (
        df.groupby("team")[existing_cols]
        .rolling(window, closed='left', min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    
    new_cols = [f"{c}_rolling" for c in existing_cols]
    rolling_stats.columns = new_cols
    
    df = pd.concat([df, rolling_stats], axis=1)
    
    return df, new_cols

def add_opponent_stats(df: pd.DataFrame) -> Tuple[pd.DataFrame, list]:
    """Merges the opponent's rolling stats onto the current row via self-join."""
    rolling_cols = [c for c in df.columns if "_rolling" in c and "_opp" not in c]
    
    if not rolling_cols:
        return df, []

    df_opp = df[["date", "team"] + rolling_cols].copy()
    rename_map = {c: f"{c}_opp" for c in rolling_cols}
    rename_map["team"] = "opponent"
    df_opp = df_opp.rename(columns=rename_map)
    
    merged = df.merge(df_opp, on=["date", "opponent"], how="left")
    
    return merged, list(rename_map.values())