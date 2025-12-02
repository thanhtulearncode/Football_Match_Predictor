import pytest
import pandas as pd
from src.pipeline.processing import clean_data
from src.pipeline.features import add_rolling_features

# Mock Data Fixture
@pytest.fixture
def mock_raw_data():
    return pd.DataFrame({
        "date": ["2023-01-01", "2023-01-08", "2023-01-15", "2023-01-22"],
        "team": ["Arsenal", "Arsenal", "Arsenal", "Arsenal"],
        "opponent": ["Chelsea", "Spurs", "Man Utd", "Everton"],
        "result": ["W", "D", "L", "W"],
        "gf": [3, 1, 0, 2],
        "ga": [1, 1, 3, 0],
        "sh": [10, 5, 2, 8],
        "sot": [5, 2, 1, 4],
        "dist": [15, 18, 20, 16],
        "fk": [1, 0, 0, 1],
        "pk": [0, 0, 0, 1],
        "pkatt": [0, 0, 0, 1]
    })

def test_clean_data_validation(mock_raw_data):
    """Ensure clean_data converts result W/D/L to 2/1/0 correctly"""
    df = clean_data(mock_raw_data)
    assert "target" in df.columns
    assert df.iloc[0]["target"] == 2 # W
    assert df.iloc[2]["target"] == 0 # L

def test_rolling_features_logic(mock_raw_data):
    """Ensure rolling averages calculate based on PREVIOUS games (closed='left')"""
    # Fix dates
    mock_raw_data["date"] = pd.to_datetime(mock_raw_data["date"])
    
    df, cols = add_rolling_features(mock_raw_data)
    
    # The first row should have NaN rolling stats (no history)
    assert pd.isna(df.iloc[0]["gf_rolling"])
    
    # The 4th row's rolling GF should be average of first 3 games: (3+1+0)/3 = 1.33
    expected_gf = (3 + 1 + 0) / 3
    actual_gf = df.iloc[3]["gf_rolling"]
    
    assert abs(actual_gf - expected_gf) < 0.01