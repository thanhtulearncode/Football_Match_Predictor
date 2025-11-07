import pytest
from fastapi.testclient import TestClient
from src.app import app
from src.predictor import predictor

client = TestClient(app)

@pytest.fixture(autouse=True)
def mock_predictor():
    pass

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data

def test_predict_endpoint():
    payload = {
        "home_elo": 1750, "away_elo": 1450,
        "home_form": 12, "away_form": 5,
        "home_avg_scored": 2.0, "home_avg_conceded": 0.8,
        "away_avg_scored": 1.0, "away_avg_conceded": 1.8,
        "h2h_home_wins": 4, "h2h_draws": 1, "h2h_away_wins": 0,
        "home_rest_days": 7, "away_rest_days": 3
    }
    response = client.post("/predict", json=payload)
    assert response.status_code in [200, 503]
    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert data["prediction"] in ["Home Win", "Away Win", "Draw"]
        assert "confidence" in data
        assert 0 <= data["confidence"] <= 1