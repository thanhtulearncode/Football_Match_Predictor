import pytest
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "model_loaded" in data
    assert data["model_loaded"] == True

def test_predict_endpoint(client):
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

def test_predict_teams_success(client):
    payload = {
        "home_team": "Arsenal FC",
        "away_team": "Liverpool FC",
        "home_rest_days": 7,
        "away_rest_days": 7
    }
    response = client.post("/predict/teams", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["home_team"] == "Arsenal FC"
    assert "prediction" in data

def test_predict_teams_invalid_team(client):
    payload = {
        "home_team": "Arsenal FC",
        "away_team": "Fake Team United", # Invalid team
        "home_rest_days": 7,
        "away_rest_days": 7
    }
    response = client.post("/predict/teams", json=payload)
    # API return 404 
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "Could not generate features" in data["detail"]

def test_get_teams_list(client):
    response = client.get("/teams")
    assert response.status_code == 200
    data = response.json()
    assert "teams" in data
    assert "count" in data
    assert data["count"] > 0
    assert "Manchester City FC" in data["teams"] 

def test_get_team_stats_invalid(client):
    response = client.get("/teams/NonExistentTeam/stats")
    assert response.status_code == 404

