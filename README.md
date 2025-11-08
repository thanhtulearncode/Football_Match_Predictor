# ⚽ Football Match Predictor

A machine learning system for predicting football match outcomes with team-based predictions and upcoming match analysis.

## 🚀 Features

- **Team-based Predictions**: Predict matches by entering team names
- **Multiple ML Models**: RandomForest, XGBoost, LightGBM, GradientBoosting, LogisticRegression
- **REST API**: Full programmatic access via FastAPI
- **Web Dashboard**: Interactive Streamlit interface
- **Advanced Analytics**: Elo ratings, form, goal statistics, head-to-head records
- **Batch Predictions**: Predict multiple matches at once

## 🛠️ Quick Start

### Installation

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Setup & Run

```bash
# 1. Set up environment variables (create .env file)
FOOTBALL_DATA_API_KEY=your_api_key_here

# 2. Fetch data
python src/data_fetcher.py

# 3. Preprocess and train models
python src/preprocess.py
python src/train.py

# 4. Start API server
uvicorn src.app:app --reload

# 5. Launch web dashboard (in new terminal)
streamlit run src/dashboard.py
```

## 📡 API Endpoints

**Core:**
- `GET /health` - Health check
- `GET /model/info` - Model information
- `GET /model/feature_importance` - Feature importance

**Predictions:**
- `POST /predict/teams` - Predict by team names (recommended)
- `POST /predict` - Predict with manual features
- `POST /predict/batch` - Batch predictions
- `GET /predict/upcoming` - Predict all upcoming matches

**Teams & Data:**
- `GET /teams` - List available teams
- `GET /teams/{team_name}/stats` - Get team statistics
- `GET /upcoming-matches` - Get match schedule

## 🎮 Usage Examples

### Team-based Prediction (Recommended)

```python
import requests

response = requests.post("http://localhost:8000/predict/teams", json={
    "home_team": "Manchester United",
    "away_team": "Liverpool",
    "home_rest_days": 7,
    "away_rest_days": 7
})
print(response.json())
```

### Manual Feature Prediction

```python
import requests

response = requests.post("http://localhost:8000/predict", json={
    "home_elo": 1650, "away_elo": 1580,
    "home_form": 10, "away_form": 7,
    "home_avg_scored": 1.8, "home_avg_conceded": 1.0,
    "away_avg_scored": 1.3, "away_avg_conceded": 1.5,
    "h2h_home_wins": 3, "h2h_draws": 1, "h2h_away_wins": 1,
    "home_rest_days": 7, "away_rest_days": 7
})
print(response.json())
```

## 📊 Dashboard

The Streamlit dashboard provides:
- Team selection and predictions
- Upcoming match analysis
- Team statistics and analytics
- Model performance metrics
- Feature importance visualization

## 🔧 Configuration

Create a `.env` file:

```env
FOOTBALL_DATA_API_KEY=your_api_key_here
API_HOST=127.0.0.1
API_PORT=8000
```

### Supported Competitions

- Premier League (PL)
- More leagues can be added via configuration

## 🏗️ System Architecture

```
Data Sources → Feature Engineering → ML Models → API → Dashboard
    ↓              ↓                  ↓         ↓        ↓
 Football     Elo Ratings,        Ensemble   FastAPI  Streamlit
 Data API     Form, H2H, etc.     Models            Interactive UI
```

## 📈 Model Performance

- **Accuracy**: ~55% (beats random chance of 33%)
- **Models**: 5 algorithms with automatic best model selection
- **Features**: 25+ engineered features (Elo ratings, form, goals, head-to-head, rest days, win rates)

## 🏗️ Project Structure

```
src/
├── app.py                 # FastAPI application
├── dashboard.py           # Streamlit dashboard
├── data_fetcher.py        # Data collection
├── preprocess.py          # Feature engineering
├── train.py              # Model training
├── predictor.py          # Prediction logic
└── config.py            # Configuration
```

## 🆘 Troubleshooting

- **No teams available**: Run `preprocess.py` after data fetching
- **API connection errors**: Ensure all services are running
- **Prediction errors**: Check team names are in available list
- **Health check**: `GET /health`

## 📄 License

This project is for educational purposes. Please ensure compliance with data source terms of service.

---

**Happy Predicting!** ⚽🎯
