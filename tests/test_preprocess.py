import pandas as pd
import pytest
from src.preprocess import (
    normalize_team_name, 
    calculate_head_to_head, 
    calculate_rest_days
)
from src.config import TEAM_NAME_MAPPINGS

def test_normalize_team_name():
    assert normalize_team_name('Man United') == 'Manchester United'
    assert normalize_team_name('Man City') == 'Manchester City'
    assert normalize_team_name('Wolves') == 'Wolverhampton Wanderers'
    assert normalize_team_name('Arsenal FC') == 'Arsenal FC'
    assert normalize_team_name('Liverpool') == 'Liverpool'

@pytest.fixture
def sample_match_data():
    data = {
        'date': pd.to_datetime([
            '2023-01-01', 
            '2023-01-10', 
            '2023-01-15', 
            '2023-01-20'
        ]),
        'home_team': ['Team A', 'Team B', 'Team A', 'Team C'],
        'away_team': ['Team B', 'Team A', 'Team C', 'Team A'],
        'home_goals': [1, 1, 0, 2], 
        'away_goals': [0, 1, 1, 0], 
        'result': [2, 1, 0, 2]  
    }
    return pd.DataFrame(data)

def test_calculate_h2h(sample_match_data):
    df = sample_match_data
    current_date = pd.to_datetime('2023-02-01') 
    h2h_A_vs_B = calculate_head_to_head(df, 'Team A', 'Team B', current_date)
    assert h2h_A_vs_B['h2h_home_wins'] == 1
    assert h2h_A_vs_B['h2h_draws'] == 1
    assert h2h_A_vs_B['h2h_away_wins'] == 0  
    assert h2h_A_vs_B['h2h_total_matches'] == 2

    h2h_A_vs_C = calculate_head_to_head(df, 'Team A', 'Team C', current_date)
    assert h2h_A_vs_C['h2h_home_wins'] == 0 
    assert h2h_A_vs_C['h2h_draws'] == 0
    assert h2h_A_vs_C['h2h_away_wins'] == 2 
    assert h2h_A_vs_C['h2h_total_matches'] == 2

    h2h_B_vs_C = calculate_head_to_head(df, 'Team B', 'Team C', current_date)
    assert h2h_B_vs_C['h2h_total_matches'] == 0

def test_calculate_rest_days(sample_match_data):
    df = sample_match_data
    
    current_date_1 = pd.to_datetime('2023-01-25')
    rest_days_A = calculate_rest_days(df, 'Team A', current_date_1)
    assert rest_days_A == 5 

    current_date_2 = pd.to_datetime('2023-01-31')
    rest_days_B = calculate_rest_days(df, 'Team B', current_date_2)
    assert rest_days_B == 21 

    rest_days_D = calculate_rest_days(df, 'Team D', current_date_1)
    assert rest_days_D == 7 # Default value