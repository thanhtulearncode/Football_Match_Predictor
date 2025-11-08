"""Streamlit dashboard for football predictions"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
from typing import Dict, List, Optional
import requests
import json
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Configure page
st.set_page_config(
    page_title="Football Match Predictor", 
    page_icon="⚽", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE = "http://127.0.0.1:8000"

@st.cache_data(ttl=300)
def load_teams():
    """Load available teams from API"""
    try:
        response = requests.get(f"{API_BASE}/teams")
        if response.status_code == 200:
            data = response.json()
            return data.get("teams", [])
    except Exception as e:
        st.error(f"Error loading teams: {e}")
    return []

@st.cache_data(ttl=300)
def load_team_stats(team_name: str):
    """Load statistics for a specific team"""
    try:
        response = requests.get(f"{API_BASE}/teams/{team_name}/stats")
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def predict_teams(home_team: str, away_team: str, home_rest: int = 7, away_rest: int = 7):
    """Predict match by team names"""
    try:
        payload = {
            "home_team": home_team,
            "away_team": away_team,
            "home_rest_days": home_rest,
            "away_rest_days": away_rest
        }
        response = requests.post(f"{API_BASE}/predict/teams", json=payload)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Prediction error: {e}")
    return None

def predict_manual(
    home_elo: float, away_elo: float,
    home_form: float, away_form: float,
    home_avg_scored: float, home_avg_conceded: float,
    away_avg_scored: float, away_avg_conceded: float,
    h2h_home_wins: int, h2h_draws: int, h2h_away_wins: int,
    home_rest_days: int, away_rest_days: int
):
    """Predict match with manual features"""
    try:
        payload = {
            "home_elo": home_elo,
            "away_elo": away_elo,
            "home_form": home_form,
            "away_form": away_form,
            "home_avg_scored": home_avg_scored,
            "home_avg_conceded": home_avg_conceded,
            "away_avg_scored": away_avg_scored,
            "away_avg_conceded": away_avg_conceded,
            "h2h_home_wins": h2h_home_wins,
            "h2h_draws": h2h_draws,
            "h2h_away_wins": h2h_away_wins,
            "home_rest_days": home_rest_days,
            "away_rest_days": away_rest_days
        }
        response = requests.post(f"{API_BASE}/predict", json=payload)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Prediction error: {e}")
    return None

@st.cache_data(ttl=600)
def load_upcoming_matches():
    """Load upcoming matches"""
    try:
        response = requests.get(f"{API_BASE}/upcoming-matches")
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

@st.cache_data(ttl=600)
def predict_upcoming_matches():
    """Predict upcoming matches"""
    try:
        response = requests.get(f"{API_BASE}/predict/upcoming")
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        st.error(f"Error predicting upcoming matches: {e}")
    return None

def create_probability_chart(probabilities: Dict) -> go.Figure:
    """Create probability visualization"""
    outcome_names = {
        'away_win': 'Away Win',
        'draw': 'Draw', 
        'home_win': 'Home Win'
    }
    
    colors = ['#EF553B', '#636EFA', '#00CC96']
    
    fig = go.Figure(go.Bar(
        x=list(probabilities.values()),
        y=[outcome_names[k] for k in probabilities.keys()],
        orientation='h',
        marker_color=colors,
        text=[f'{p:.1%}' for p in probabilities.values()],
        textposition='auto',
        textfont=dict(size=14, color='white')
    ))
    
    fig.update_layout(
        title=dict(text="Probability Distribution", x=0.5, xanchor='center'),
        height=250,
        showlegend=False,
        xaxis=dict(title="Probability", range=[0, 1]),
        yaxis=dict(title="Outcome"),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

def display_team_stats(team_name: str):
    """Display team statistics"""
    stats_data = load_team_stats(team_name)
    if not stats_data:
        st.warning(f"Statistics not available for {team_name}")
        return
    
    stats = stats_data['stats']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Elo Rating", f"{stats.get('elo', 1500):.0f}")
    with col2:
        st.metric("Current Form", f"{stats.get('form', 0):.0f}")
    with col3:
        st.metric("Avg Goals Scored", f"{stats.get('avg_scored', 1.5):.2f}")
    with col4:
        st.metric("Avg Goals Conceded", f"{stats.get('avg_conceded', 1.5):.2f}")

def main():
    st.title("⚽ Football Match Predictor")
    
    # Sidebar
    with st.sidebar:
        st.title("Navigation")
        page = st.radio(
            "Choose Page",
            ["Team Prediction", "Upcoming Matches", "Manual Prediction", "Analysis", "Model Info"]
        )
        
        # API status
        try:
            health_response = requests.get(f"{API_BASE}/health")
            if health_response.status_code == 200:
                health_data = health_response.json()
                st.success(f"✅ API: {health_data['status']}")
                st.info(f"🤖 Model: {health_data.get('model_name', 'Unknown')}")
                st.info(f"🏃 Teams: {health_data.get('teams_loaded', 0)}")
                st.info(f"📅 Upcoming: {health_data.get('upcoming_matches_loaded', 0)}")
            else:
                st.error("❌ API not responding")
        except:
            st.error("❌ Cannot connect to API")

    # Main content
    if page == "Team Prediction":
        render_team_prediction()
    elif page == "Upcoming Matches":
        render_upcoming_matches()
    elif page == "Manual Prediction":
        render_manual_prediction()
    elif page == "Analysis":
        render_analysis_page()
    elif page == "Model Info":
        render_model_info()

def render_team_prediction():
    """Render team-based prediction interface"""
    st.header("🏟️ Predict by Team Names")
    
    # Load teams
    teams = load_teams()
    
    if not teams:
        st.error("No teams available. Please ensure the API is running and data is processed.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏠 Home Team")
        home_team = st.selectbox("Select Home Team", teams, key="home_team_select")
        home_rest = st.slider("Home Team Rest Days", 0, 30, 7, key="home_rest")
        
        if home_team:
            with st.expander(f"📊 {home_team} Statistics"):
                display_team_stats(home_team)

    with col2:
        st.subheader("✈️ Away Team")
        away_team = st.selectbox("Select Away Team", teams, key="away_team_select")
        away_rest = st.slider("Away Team Rest Days", 0, 30, 7, key="away_rest")
        
        if away_team:
            with st.expander(f"📊 {away_team} Statistics"):
                display_team_stats(away_team)
    
    # Quick predictions for popular matches
    st.subheader("🚀 Quick Predictions")
    popular_matches = [
        ("Manchester United", "Liverpool"),
        ("Arsenal", "Chelsea"),
        ("Manchester City", "Tottenham Hotspur"),
        ("Newcastle United", "Brighton & Hove Albion"),
        ("West Ham United", "AFC Bournemouth")
    ]
    
    # Filter available popular matches
    available_popular_matches = [(h, a) for h, a in popular_matches if h in teams and a in teams]
    
    if available_popular_matches:
        cols = st.columns(len(available_popular_matches))
        for idx, (home, away) in enumerate(available_popular_matches):
            with cols[idx]:
                if st.button(f"{home}\nvs\n{away}", use_container_width=True, key=f"quick_{idx}"):
                    result = predict_teams(home, away)
                    if result:
                        display_prediction_result(result)
    
    # Main prediction button
    if st.button("🎯 Predict Match", type="primary", use_container_width=True):
        if home_team == away_team:
            st.error("Please select different teams")
        else:
            with st.spinner("Analyzing teams and generating prediction..."):
                result = predict_teams(home_team, away_team, home_rest, away_rest)
            
            if result:
                display_prediction_result(result)

def render_upcoming_matches():
    """Render upcoming matches prediction interface"""
    st.header("📅 Upcoming Matches Predictions")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create a container to hold the predictions
        predictions_container = st.container()
        
        if st.button("🔄 Load & Predict Upcoming Matches", type="primary"):
            with st.spinner("Loading and predicting upcoming matches..."):
                predictions_data = predict_upcoming_matches()
                
            if predictions_data and predictions_data.get('predictions'):
                predictions = predictions_data['predictions']
                st.success(f"✅ Predicted {len(predictions)} upcoming matches")
                
                # Display predictions in the dedicated container
                with predictions_container:
                    for pred in predictions:
                        with st.container():
                            pred_col1, pred_col2, pred_col3 = st.columns([3, 2, 1])
                            
                            with pred_col1:
                                st.write(f"**{pred['home_team']}** vs **{pred['away_team']}**")
                                st.caption(f"Date: {pred['date']} | Competition: {pred.get('competition', 'Unknown')}")
                            
                            with pred_col2:
                                outcome_emoji = {"Home Win": "🏠", "Draw": "🤝", "Away Win": "✈️"}
                                st.metric(
                                    "Prediction", 
                                    f"{outcome_emoji.get(pred['prediction'], '⚽')} {pred['prediction']}"
                                )
                            
                            with pred_col3:
                                st.metric("Confidence", f"{pred['confidence']:.1%}")
                                
                                # Show probabilities
                                probs = pred['probabilities']
                                st.progress(int(probs['home_win'] * 100), text=f"Home: {probs['home_win']:.1%}")
                                st.progress(int(probs['draw'] * 100), text=f"Draw: {probs['draw']:.1%}")
                                st.progress(int(probs['away_win'] * 100), text=f"Away: {probs['away_win']:.1%}")
                            
                            st.divider()
            else:
                st.warning("No upcoming matches predictions available")
    
    with col2:
        st.subheader("Custom Upcoming Matches")
        st.info("Add custom matches to predict")
        
        with st.form("custom_match_form"):
            home_team = st.text_input("Home Team")
            away_team = st.text_input("Away Team")
            match_date = st.date_input("Match Date")
            competition = st.text_input("Competition (optional)")
            
            if st.form_submit_button("Add & Predict Custom Match"):
                if home_team and away_team:
                    # This would call the custom prediction endpoint
                    st.info("Custom match prediction would be implemented here")
                else:
                    st.warning("Please enter both teams")

def render_manual_prediction():
    """Render manual feature input prediction"""
    st.header("🎯 Manual Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏠 Home Team")
        home_elo = st.slider("Elo Rating", 1000, 2500, 1650, key="home_elo")
        home_form = st.slider("Form Points", 0, 15, 10, key="home_form")
        home_avg_scored = st.number_input("Avg Goals Scored", 0.0, 5.0, 1.8, 0.1, key="home_scored")
        home_avg_conceded = st.number_input("Avg Goals Conceded", 0.0, 5.0, 1.0, 0.1, key="home_conceded")
        home_rest = st.number_input("Rest Days", 0, 30, 7, key="home_rest")

    with col2:
        st.subheader("✈️ Away Team")
        away_elo = st.slider("Elo Rating", 1000, 2500, 1580, key="away_elo")
        away_form = st.slider("Form Points", 0, 15, 7, key="away_form")
        away_avg_scored = st.number_input("Avg Goals Scored", 0.0, 5.0, 1.3, 0.1, key="away_scored")
        away_avg_conceded = st.number_input("Avg Goals Conceded", 0.0, 5.0, 1.5, 0.1, key="away_conceded")
        away_rest = st.number_input("Rest Days", 0, 30, 7, key="away_rest")

    # Head-to-head
    st.subheader("🤝 Head-to-Head Statistics")
    h2h_col1, h2h_col2, h2h_col3 = st.columns(3)
    with h2h_col1:
        h2h_home = st.number_input("Home Team Wins", 0, 20, 3, key="h2h_home")
    with h2h_col2:
        h2h_draw = st.number_input("Draws", 0, 20, 1, key="h2h_draw")
    with h2h_col3:
        h2h_away = st.number_input("Away Team Wins", 0, 20, 1, key="h2h_away")

    if st.button("🚀 Predict Match", type="primary", use_container_width=True):
        # Validate head-to-head totals
        h2h_total = h2h_home + h2h_draw + h2h_away
        if h2h_total > 20:
            st.error("Total head-to-head matches cannot exceed 20")
        else:
            with st.spinner("Analyzing features and generating prediction..."):
                result = predict_manual(
                    home_elo=home_elo,
                    away_elo=away_elo,
                    home_form=home_form,
                    away_form=away_form,
                    home_avg_scored=home_avg_scored,
                    home_avg_conceded=home_avg_conceded,
                    away_avg_scored=away_avg_scored,
                    away_avg_conceded=away_avg_conceded,
                    h2h_home_wins=h2h_home,
                    h2h_draws=h2h_draw,
                    h2h_away_wins=h2h_away,
                    home_rest_days=int(home_rest),
                    away_rest_days=int(away_rest)
                )
            
            if result:
                display_prediction_result(result)

def render_analysis_page():
    """Render data analysis page"""
    st.header("📈 Data Analysis")
    
    # Load teams for analysis
    teams = load_teams()
    
    if teams:
        selected_team = st.selectbox("Select Team for Analysis", teams)
        if selected_team:
            stats_data = load_team_stats(selected_team)
            if stats_data:
                stats = stats_data['stats']
                
                # Create metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Elo Rating", f"{stats.get('elo', 1500):.0f}")
                with col2:
                    st.metric("Current Form", f"{stats.get('form', 0):.0f}/15")
                with col3:
                    st.metric("Win Rate", f"{stats.get('win_rate', 0.33):.1%}")
                with col4:
                    st.metric("Matches Played", stats.get('total_matches', 0))
                
                # Goal statistics
                st.subheader("Goal Statistics")
                goal_data = {
                    'Metric': ['Avg Scored', 'Avg Conceded', 'Goal Difference'],
                    'Value': [
                        stats.get('avg_scored', 1.5),
                        stats.get('avg_conceded', 1.5),
                        stats.get('goal_difference', 0)
                    ]
                }
                st.bar_chart(pd.DataFrame(goal_data).set_index('Metric'))
    
    else:
        st.warning("No data available for analysis")

def render_model_info():
    """Render model information page"""
    st.header("🤖 Model Information")
    
    try:
        # Get model info
        response = requests.get(f"{API_BASE}/model/info")
        if response.status_code == 200:
            model_info = response.json()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Model Name", model_info.get('model_name', 'Unknown'))
            with col2:
                st.metric("Model Type", model_info.get('model_type', 'Unknown'))
            with col3:
                st.metric("Features", model_info.get('n_features', 0))
            with col4:
                st.metric("Has Feature Importance", "Yes" if model_info.get('has_feature_importance') else "No")
            
            # Feature importance
            if model_info.get('has_feature_importance'):
                st.subheader("Feature Importance")
                importance_response = requests.get(f"{API_BASE}/model/feature_importance")
                if importance_response.status_code == 200:
                    importance_data = importance_response.json()
                    importance_df = pd.DataFrame(importance_data['feature_importance'])
                    
                    fig = px.bar(
                        importance_df.head(10),
                        x='importance',
                        y='feature',
                        orientation='h',
                        title="Top 10 Most Important Features",
                        color='importance'
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.error("Could not load model information")
            
    except Exception as e:
        st.error(f"Error loading model info: {e}")

def display_prediction_result(result: Dict):
    """Display prediction results"""
    st.success(f"## **Prediction: {result['prediction']}**")
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Confidence", f"{result['confidence']:.1%}")
    with col2:
        st.metric("Model Used", result['model_used'])
    with col3:
        st.metric("Prediction Code", result['prediction_code'])
    with col4:
        outcome_map = {"Home Win": "🏠", "Draw": "🤝", "Away Win": "✈️"}
        st.metric("Outcome", outcome_map.get(result['prediction'], "⚽"))
    
    # Probability chart
    st.plotly_chart(create_probability_chart(result['probabilities']), use_container_width=True)
    
    # Feature details
    with st.expander("📊 View Feature Details"):
        if 'features_used' in result:
            features = result['features_used']
            feature_df = pd.DataFrame({
                'Feature': list(features.keys()),
                'Value': list(features.values())
            })
            st.dataframe(feature_df, use_container_width=True)

if __name__ == "__main__":
    main()