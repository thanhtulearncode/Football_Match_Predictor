# PL Oracle: Football Match Predictor

A Machine Learning application that predicts Premier League match outcomes. It features a FastAPI backend for serving predictions and a modern React (Vite) dashboard for visualization.

## Architecture

- **Backend**: Python, FastAPI, Scikit-Learn/XGBoost
- **Frontend**: React, Vite, Tailwind CSS
- **Data**: Scraped from FBref (Pandas, BeautifulSoup),
- **Deployment**: Docker & Docker Compose

## Quick Start (Local Development)

To run the project locally, you need two terminal windows open simultaneously (one for the API, one for the Web Interface).

### Setup Prerequisites

**Backend (Python):**

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Frontend (Node.js):**

```bash
cd web
npm install
cd ..
```

### Run the Application

**Terminal A: The Brain (Backend)**
This starts the Prediction API.

```bash
# Make sure venv is active
python -m uvicorn src.api.app:app --reload --port 8080
```

API will run at: http://127.0.0.1:8080

**Terminal B: The Face (Frontend)**
This starts the Dashboard.

```bash
cd web
npm run dev
```

Dashboard will run at: http://localhost:3000

> **Note:** The dashboard automatically detects your API address. If you open the dashboard on localhost, it talks to localhost:8000. If you open it on 127.0.0.1, it talks to 127.0.0.1:8000.

## Quick Start (Docker Production)

If you want to run the exact system that would run on a server (simulating production), use Docker. This requires no manual dependency installation.

```bash
# Build and Start
docker-compose up --build
```

- Web Dashboard: http://localhost:3000
- API Docs: http://localhost:8000/docs

## ML Pipeline Operations

The prediction model relies on up-to-date data. Run the pipeline periodically to scrape new match results and retrain the model.

**Run the full pipeline (Scrape → Process → Train):**

```bash
python scripts/run_pipeline.py
```

**Check specific team predictions via CLI:**

```bash
python scripts/show_predictions.py
```

## Project Structure

```
├── data/              # CSV Storage
│   ├── raw/          # Scraped data
│   └── processed/    # Cleaned features
├── models/           # Saved .pkl models & encoders
├── scripts/          # Automation scripts (pipeline)
├── src/              # Python Source Code
│   ├── api/          # FastAPI Endpoints
│   └── pipeline/     # ML Logic (Scraping, Feature Eng, Training)
└── web/              # React Application
    ├── src/          # React Components
    └── public/       # Static Assets
```

## Troubleshooting

### 1. "Connection Error" in Dashboard

- Ensure Terminal A (Backend) is actually running.
- Check if the backend port (8000) is blocked.
- Ensure you aren't mixing localhost and 127.0.0.1.

### 2. 404 Not Found on Frontend

- Ensure `web/index.html` exists in the root of the web folder.
- Ensure you are running `npm run dev` from inside the `web/` directory.

### 3. "FutureWarning: Importing pandas-specific..."

This is a harmless warning from the pandera library in the backend logs. It does not affect functionality.
