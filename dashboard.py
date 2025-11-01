"""Streamlit dashboard for football predictions"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import joblib

st.set_page_config(page_title="Football Match Predictor", page_icon="⚽", layout="wide")
MODELS_DIR = Path("models")
PROCESSED_DIR = Path("data") / "processed"

@st.cache_resource
def load_model():
    """Load best model"""
    try:
        comparison_path = MODELS_DIR / "model_comparison.csv"
        if comparison_path.exists():
            comparison = pd.read_csv(comparison_path, index_col=0)
            best_model_name = comparison['test_accuracy'].idxmax()
            model_path = MODELS_DIR / f"{best_model_name}_model.pkl"
            if model_path.exists():
                return joblib.load(model_path), best_model_name
        st.error("No model found. Run train.py first.")
        return None, None
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None

@st.cache_data
def load_historical_data():
    """Load historical data"""
    try:
        train_path = PROCESSED_DIR / "train.csv"
        if train_path.exists():
            return pd.read_csv(train_path, parse_dates=['date'])
        return pd.DataFrame()
    except:
        return pd.DataFrame()

def predict_match(model, features):
    """Make prediction"""
    try:
        df = pd.DataFrame([features])
        prediction = int(model.predict(df)[0])
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(df)[0]
            probabilities = {"Away Win": float(proba[0]), "Draw": float(proba[1]), "Home Win": float(proba[2])}
        else:
            probabilities = {"Away Win": 1.0 if prediction == 0 else 0.0,
                           "Draw": 1.0 if prediction == 1 else 0.0,
                           "Home Win": 1.0 if prediction == 2 else 0.0}
        outcome_map = {0: "Away Win", 1: "Draw", 2: "Home Win"}
        return outcome_map[prediction], probabilities
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None

def main():
    st.title("⚽ Football Match Predictor")
    model, model_name = load_model()
    if not model:
        st.stop()
    with st.sidebar:
        st.info(f"Model: {model_name}")
        page = st.radio("Page", ["Prediction", "Analysis", "Model Info"])
    
    if page == "Prediction":
        st.header("Match Prediction")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Home Team")
            home_elo = st.slider("Elo", 1000, 2500, 1650)
            home_form = st.slider("Form", 0, 15, 10)
            home_avg_scored = st.number_input("Avg Scored", 0.0, 5.0, 1.8, 0.1)
            home_avg_conceded = st.number_input("Avg Conceded", 0.0, 5.0, 1.0, 0.1)
            home_rest = st.number_input("Rest Days", 0, 30, 7)
        with col2:
            st.subheader("Away Team")
            away_elo = st.slider("Elo", 1000, 2500, 1580, key="away_elo")
            away_form = st.slider("Form", 0, 15, 7, key="away_form")
            away_avg_scored = st.number_input("Avg Scored", 0.0, 5.0, 1.3, 0.1, key="away_scored")
            away_avg_conceded = st.number_input("Avg Conceded", 0.0, 5.0, 1.5, 0.1, key="away_conceded")
            away_rest = st.number_input("Rest Days", 0, 30, 7, key="away_rest")
        h2h_col1, h2h_col2, h2h_col3 = st.columns(3)
        with h2h_col1:
            h2h_home = st.number_input("H2H Home Wins", 0, 20, 3)
        with h2h_col2:
            h2h_draw = st.number_input("H2H Draws", 0, 20, 1)
        with h2h_col3:
            h2h_away = st.number_input("H2H Away Wins", 0, 20, 1)
        
        features = {
            'home_elo': home_elo, 'away_elo': away_elo, 'elo_diff': home_elo - away_elo,
            'home_form': home_form, 'away_form': away_form, 'form_diff': home_form - away_form,
            'home_avg_scored': home_avg_scored, 'home_avg_conceded': home_avg_conceded,
            'away_avg_scored': away_avg_scored, 'away_avg_conceded': away_avg_conceded,
            'attack_strength_diff': home_avg_scored - away_avg_conceded,
            'h2h_home_wins': h2h_home, 'h2h_draws': h2h_draw, 'h2h_away_wins': h2h_away,
            'home_rest_days': home_rest, 'away_rest_days': away_rest,
            'rest_advantage': home_rest - away_rest
        }
        
        if st.button("Predict", type="primary"):
            prediction, probabilities = predict_match(model, features)
            if prediction:
                st.success(f"Prediction: {prediction}")
                st.metric("Confidence", f"{max(probabilities.values()):.1%}")
                fig = go.Figure(go.Bar(x=list(probabilities.values()), y=list(probabilities.keys()),
                                     orientation='h', marker_color=['#4facfe', '#f5576c', '#667eea']))
                fig.update_layout(title="Probabilities", height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Analysis":
        st.header("Historical Analysis")
        df = load_historical_data()
        if df.empty:
            st.warning("No data available")
        else:
            st.metric("Total Matches", len(df))
            outcome_counts = df['target'].value_counts()
            fig = px.pie(values=outcome_counts.values, names=['Away', 'Draw', 'Home'],
                        title="Outcome Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Model Info":
        st.header("Model Information")
        st.info(f"Model: {model_name}")
        if hasattr(model, 'feature_importances_'):
            feature_names = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else \
                           [f"feature_{i}" for i in range(len(model.feature_importances_))]
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False).head(15)
            fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                        title="Top 15 Features")
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
