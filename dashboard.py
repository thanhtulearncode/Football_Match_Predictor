"""Streamlit dashboard for football predictions"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
from typing import Dict, Optional, Tuple
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Configure page first
st.set_page_config(
    page_title="Football Match Predictor", 
    page_icon="⚽", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache configuration (cache resources for 1 hour to avoid reloading)
@st.cache_resource(show_spinner=False, ttl=3600)
def load_predictor():
    """Load the ML predictor model (cached for performance)"""
    try:
        from src.predictor import predictor
        success = predictor.load_best_model()
        if success:
            st.success("✅ Model loaded successfully")
            return predictor
        else:
            st.error("❌ Failed to load model")
            return None
    except Exception as e:
        st.error(f"❌ Error loading predictor: {e}")
        return None

# Cache data for 10 minutes
@st.cache_data(show_spinner=False, ttl=600)
def load_historical_data() -> pd.DataFrame:
    """Load and cache historical match data"""
    try:
        from src.config import PROCESSED_DIR
        train_path = PROCESSED_DIR / "train.csv"
        if train_path.exists():
            df = pd.read_csv(train_path, parse_dates=['date'])
            st.success(f"✅ Loaded {len(df)} historical matches")
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def get_model_info(predictor_instance) -> Dict:
    """Get model information"""
    if not predictor_instance or not predictor_instance.model:
        return {}
    
    info = {
        "model_name": getattr(predictor_instance, 'model_name', 'Unknown'),
        "model_type": type(predictor_instance.model).__name__,
        "n_features": len(predictor_instance.feature_names) if predictor_instance.feature_names else 0,
        "has_feature_importance": hasattr(predictor_instance.model, 'feature_importances_')
    }
    
    # Add model-specific attributes
    model_attrs = ['n_estimators', 'max_depth', 'learning_rate']
    for attr in model_attrs:
        if hasattr(predictor_instance.model, attr):
            info[attr] = getattr(predictor_instance.model, attr)
    
    return info

def calculate_derived_features(input_features: Dict) -> Dict:
    """Calculate derived features from input (differences between teams)"""
    return {
        **input_features,
        'elo_diff': input_features['home_elo'] - input_features['away_elo'],  # Elo difference
        'form_diff': input_features['home_form'] - input_features['away_form'],  # Form difference
        'attack_strength_diff': input_features['home_avg_scored'] - input_features['away_avg_conceded'],  # Attack strength
        'rest_advantage': input_features['home_rest_days'] - input_features['away_rest_days']  # Rest advantage
    }

def predict_match(predictor_instance, input_features: Dict) -> Optional[Dict]:
    """Make match prediction using loaded model"""
    if not predictor_instance or not predictor_instance.model:
        return None
    
    try:
        # Calculate derived features and predict
        full_features = calculate_derived_features(input_features)
        return predictor_instance.predict(full_features)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

def create_probability_chart(probabilities: Dict) -> go.Figure:
    """Create probability visualization"""
    outcome_names = {
        'away_win': 'Away Win',
        'draw': 'Draw', 
        'home_win': 'Home Win'
    }
    
    colors = ['#EF553B', '#636EFA', '#00CC96']  # Red, Blue, Green
    
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
        title=dict(
            text="Probability Distribution",
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        height=250,
        showlegend=False,
        xaxis=dict(
            title="Probability",
            range=[0, max(1.0, max(probabilities.values()) * 1.1)]
        ),
        yaxis=dict(title="Outcome"),
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

def create_outcome_distribution_chart(df: pd.DataFrame) -> Tuple[go.Figure, go.Figure]:
    """Create outcome distribution charts"""
    if df.empty:
        return go.Figure(), go.Figure()
    
    outcome_counts = df['target'].value_counts().sort_index()
    outcome_names = ['Away Win', 'Draw', 'Home Win']
    colors = ['#EF553B', '#636EFA', '#00CC96']
    
    # Pie chart
    pie_fig = go.Figure(go.Pie(
        labels=outcome_names,
        values=outcome_counts.values,
        marker=dict(colors=colors),
        textinfo='percent+label',
        hole=0.3
    ))
    
    pie_fig.update_layout(
        title=dict(text="Match Outcome Distribution", x=0.5, xanchor='center'),
        height=400
    )
    
    # Bar chart
    bar_fig = go.Figure(go.Bar(
        x=outcome_names,
        y=outcome_counts.values,
        marker_color=colors,
        text=outcome_counts.values,
        textposition='auto'
    ))
    
    bar_fig.update_layout(
        title=dict(text="Match Count by Outcome", x=0.5, xanchor='center'),
        xaxis_title="Outcome",
        yaxis_title="Number of Matches",
        height=400
    )
    
    return pie_fig, bar_fig

def create_feature_importance_chart(predictor_instance) -> go.Figure:
    """Create feature importance visualization"""
    if not predictor_instance or not hasattr(predictor_instance.model, 'feature_importances_'):
        return go.Figure()
    
    # Get feature names
    if predictor_instance.feature_names:
        feature_names = predictor_instance.feature_names
    else:
        feature_names = [f"Feature_{i}" for i in range(len(predictor_instance.model.feature_importances_))]
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': predictor_instance.model.feature_importances_
    }).sort_values('Importance', ascending=True).tail(15)  # Top 15 features
    
    fig = go.Figure(go.Bar(
        x=importance_df['Importance'],
        y=importance_df['Feature'],
        orientation='h',
        marker_color='#1f77b4',
        text=[f'{imp:.3f}' for imp in importance_df['Importance']],
        textposition='auto'
    ))
    
    fig.update_layout(
        title=dict(
            text="Top 15 Feature Importances",
            x=0.5,
            xanchor='center',
            font=dict(size=16)
        ),
        height=500,
        xaxis_title="Importance",
        yaxis_title="Features",
        margin=dict(l=150, r=50, t=50, b=50)
    )
    
    return fig

def render_prediction_page(predictor_instance):
    """Render the match prediction page with input forms"""
    st.header("🎯 Match Prediction")
    
    # Use columns for better layout (home team on left, away team on right)
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

    # Head-to-head section
    st.subheader("🤝 Head-to-Head Statistics")
    h2h_col1, h2h_col2, h2h_col3 = st.columns(3)
    with h2h_col1:
        h2h_home = st.number_input("Home Team Wins", 0, 20, 3, key="h2h_home")
    with h2h_col2:
        h2h_draw = st.number_input("Draws", 0, 20, 1, key="h2h_draw")
    with h2h_col3:
        h2h_away = st.number_input("Away Team Wins", 0, 20, 1, key="h2h_away")

    # Input features
    input_features = {
        'home_elo': home_elo,
        'away_elo': away_elo,
        'home_form': home_form,
        'away_form': away_form,
        'home_avg_scored': home_avg_scored,
        'home_avg_conceded': home_avg_conceded,
        'away_avg_scored': away_avg_scored,
        'away_avg_conceded': away_avg_conceded,
        'h2h_home_wins': h2h_home,
        'h2h_draws': h2h_draw,
        'h2h_away_wins': h2h_away,
        'home_rest_days': home_rest,
        'away_rest_days': away_rest
    }

    # Prediction button
    if st.button("🚀 Predict Match Outcome", type="primary", use_container_width=True):
        with st.spinner("Analyzing match data..."):
            time.sleep(0.5)  # Small delay for better UX
            result = predict_match(predictor_instance, input_features)
        
        if result:
            # Display results
            outcome_color = {
                "Away Win": "red",
                "Draw": "blue", 
                "Home Win": "green"
            }.get(result['prediction'], "gray")
            
            st.success(f"## **Prediction: :{outcome_color}[{result['prediction']}]**")
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Confidence", f"{result['confidence']:.1%}")
            with col2:
                st.metric("Model Used", result['model_used'])
            with col3:
                st.metric("Prediction Code", result['prediction_code'])
            
            # Probability chart
            st.plotly_chart(create_probability_chart(result['probabilities']), use_container_width=True)
            
            # Feature details
            with st.expander("📊 View Feature Details"):
                derived = calculate_derived_features(input_features)
                st.json({k: round(v, 3) if isinstance(v, float) else v for k, v in derived.items()})

def render_analysis_page():
    """Render data analysis page with statistics and charts"""
    st.header("📈 Data Analysis")
    
    # Load historical match data
    df = load_historical_data()
    if df.empty:
        st.warning("No historical data available. Run preprocess.py first.")
        return
    
    # Display overview metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Matches", len(df))
    with col2:
        st.metric("Unique Teams", df['home_team'].nunique())
    with col3:
        st.metric("Date Range", f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
    with col4:
        st.metric("Features Available", len(df.columns) - 4)  # Exclude metadata columns
    
    # Charts
    st.subheader("Match Outcome Analysis")
    pie_fig, bar_fig = create_outcome_distribution_chart(df)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(pie_fig, use_container_width=True)
    with col2:
        st.plotly_chart(bar_fig, use_container_width=True)
    
    # Additional analysis
    with st.expander("📋 View Sample Data"):
        st.dataframe(df.head(100), use_container_width=True)

def render_model_info_page(predictor_instance):
    """Render model information page with details and feature importance"""
    st.header("🤖 Model Information")
    
    # Get model information
    model_info = get_model_info(predictor_instance)
    if not model_info:
        st.error("No model information available")
        return
    
    # Display model overview metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model Name", model_info.get('model_name', 'Unknown'))
    with col2:
        st.metric("Model Type", model_info.get('model_type', 'Unknown'))
    with col3:
        st.metric("Number of Features", model_info.get('n_features', 0))
    with col4:
        st.metric("Feature Importance", "Available" if model_info.get('has_feature_importance') else "Not Available")
    
    # Feature importance chart
    if model_info.get('has_feature_importance'):
        st.subheader("Feature Importance Analysis")
        importance_fig = create_feature_importance_chart(predictor_instance)
        st.plotly_chart(importance_fig, use_container_width=True)
    else:
        st.info("This model type doesn't support feature importance visualization.")
    
    # Model parameters
    st.subheader("Model Parameters")
    param_cols = [col for col in ['n_estimators', 'max_depth', 'learning_rate'] if col in model_info]
    if param_cols:
        cols = st.columns(len(param_cols))
        for idx, param in enumerate(param_cols):
            with cols[idx]:
                st.metric(param.replace('_', ' ').title(), model_info[param])
    else:
        st.info("No specific parameters available for this model type.")

def main():
    """Main application entry point with sidebar navigation"""
    # Sidebar with loading states and navigation
    with st.sidebar:
        st.title("⚽ Football Predictor")
        st.markdown("---")
        
        # Load ML model (cached)
        with st.spinner("Loading model..."):
            predictor_instance = load_predictor()
        
        # Page navigation radio buttons
        page = st.radio(
            "Navigation",
            ["🎯 Prediction", "📈 Analysis", "🤖 Model Info"],
            index=0
        )
        
        # Display model status in sidebar
        if predictor_instance:
            st.markdown("---")
            st.subheader("Model Status")
            st.success(f"**Model**: {getattr(predictor_instance, 'model_name', 'Unknown')}")
            st.info(f"**Features**: {len(predictor_instance.feature_names) if predictor_instance.feature_names else 0}")
        
        st.markdown("---")
        st.caption("Football Match Prediction Dashboard v2.0")

    # Main content area - render selected page
    try:
        if page == "🎯 Prediction":
            render_prediction_page(predictor_instance)
        elif page == "📈 Analysis":
            render_analysis_page()
        elif page == "🤖 Model Info":
            render_model_info_page(predictor_instance)
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.info("Please check if all required services are running properly.")

if __name__ == "__main__":
    main()