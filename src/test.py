"""Test script for Football Prediction API"""

import requests
from config import API_HOST, API_PORT

# API base URL
API_URL = f"http://{API_HOST}:{API_PORT}"

def test_health_check():
    """Test API health endpoint"""
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    return response.status_code == 200

def test_model_info():
    """Test model information endpoint"""
    response = requests.get(f"{API_URL}/model/info")
    if response.status_code == 200:
        info = response.json()
        print(f"Model: {info['model_name']}, Features: {info['n_features']}")
    else:
        print(f"Error: {response.status_code}")

def test_single_prediction():
    """Test single match prediction endpoint"""
    # Sample match features for testing
    payload = {
        "home_elo": 1750, "away_elo": 1450, "elo_diff": 300,
        "home_form": 12, "away_form": 5, "form_diff": 7,
        "home_avg_scored": 2.0, "home_avg_conceded": 0.8,
        "away_avg_scored": 1.0, "away_avg_conceded": 1.8,
        "attack_strength_diff": 1.0,
        "h2h_home_wins": 4, "h2h_draws": 1, "h2h_away_wins": 0,
        "home_rest_days": 7, "away_rest_days": 3, "rest_advantage": 4
    }
    response = requests.post(f"{API_URL}/predict", json=payload)
    if response.status_code == 200:
        result = response.json()
        print(f"Prediction: {result['prediction']} ({result['confidence']:.1%})")
        return True
    print(f"Error: {response.status_code}")
    return False

def run_all_tests():
    """Run all API tests"""
    # Check if API is running
    if not test_health_check():
        print("API not running. Start with: uvicorn src.app:app --reload")
        return
    # Test model info
    test_model_info()
    # Test prediction
    test_single_prediction()

if __name__ == "__main__":
    try:
        run_all_tests()
    except requests.exceptions.ConnectionError:
        print("Cannot connect to API")
    except Exception as e:
        print(f"Error: {e}")
