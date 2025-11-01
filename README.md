# Football Match Predictor

ML system for predicting football match outcomes.

## Setup

```bash
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Usage

```bash
# Fetch data
python src/data_fetcher.py

# Preprocess
python src/preprocess.py

# Train models
python src/train.py

# Run API
uvicorn src.app:app --reload

# Run dashboard
streamlit run dashboard.py
```

## API Endpoints

- `GET /health` - Health check
- `GET /model/info` - Model information
- `POST /predict` - Single prediction
- `POST /predict/batch` - Batch predictions
- `GET /model/feature_importance` - Feature importance

### Example API Call

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "home_elo": 1650, "away_elo": 1580, "elo_diff": 70,
    "home_form": 10, "away_form": 7, "form_diff": 3,
    "home_avg_scored": 1.8, "home_avg_conceded": 1.0,
    "away_avg_scored": 1.3, "away_avg_conceded": 1.5,
    "attack_strength_diff": 0.3,
    "h2h_home_wins": 3, "h2h_draws": 1, "h2h_away_wins": 1,
    "home_rest_days": 7, "away_rest_days": 7, "rest_advantage": 0
  }'
```

Python example:
```python
import requests
response = requests.post("http://localhost:8000/predict", json={
    "home_elo": 1650, "away_elo": 1580, "elo_diff": 70,
    "home_form": 10, "away_form": 7, "form_diff": 3,
    "home_avg_scored": 1.8, "home_avg_conceded": 1.0,
    "away_avg_scored": 1.3, "away_avg_conceded": 1.5,
    "attack_strength_diff": 0.3,
    "h2h_home_wins": 3, "h2h_draws": 1, "h2h_away_wins": 1,
    "home_rest_days": 7, "away_rest_days": 7, "rest_advantage": 0
})
print(response.json())
```

## Features

- Elo ratings
- Form (recent points)
- Goal statistics
- Head-to-head records
- Rest days

## Models

XGBoost, LightGBM, RandomForest, GradientBoosting, LogisticRegression
