# Future improvements on the project can be implementing ETAS -> ML method
# DBSCAN
# Let's try logistic regression for now


# To run file use: streamlit run app.py

#import streamlit as st
#st.title('Earthquake Risk Predictor')
#st.write('Setup is working!')


###############
# Optional to do

# There is a redundant function between compute_freq_series in 'features.py' and freq_plot in ''viz.py


'''
Missing tasks:
1) Implement feature scripts - done
2) Hopsworks connection - done
3) Implement the ML scripts - done 
#  Solve for optional to do list
4) Deployment
'''

import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, timezone

# CRITICAL: Load .env file BEFORE anything else
# Try multiple paths to ensure we find it
env_paths = [
    Path('.env'),  # Current directory
    Path('/app/.env'),  # Docker default
    Path(__file__).parent / '.env',  # Same directory as app.py
]

for env_path in env_paths:
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
        break
else:
    # If no .env found, try loading from current directory anyway
    load_dotenv(override=True)
from src.viz import mag_hist, map_plot, freq_plot
from src.usgs_client import get_earthquakes
from src.features import add_seq_feat, basic_time_feats, mag_stats, depth_stats
from src.labels import add_aftershock_label
from src.models import predict_aftershock_proba
from src.hopsworks_client import (
    save_features_to_hopsworks,
    load_model_from_hopsworks
)

# Variables
T_hours = 24
R_km = 100
limit_num_eartquakes = 1000
max_mag = 8.0

# Hopsworks settings
USE_HOPSWORKS = True  # Toggle Hopsworks integration


# Loading of earthquakes
@st.cache_data(ttl=60, show_spinner=False)
def load_data(starttime:str, endtime:str, min_mag:float, bbox, limit:int = 10):
    df = get_earthquakes(starttime=starttime,endtime=endtime,min_magnitude=min_mag, bbox=bbox, limit=limit)
    return df



# Let's define some regions to test
REGIONS = {
    'Japan': (122,24,146,46),
    'Mexico': (-118,14,-86,33),
    'Chile': (-76,-56,-66,-17),
    'Global': None,    
}

st.set_page_config(page_title='Earthquake Aftershock Predictor', layout='wide')
st.title('🌍 Earthquake Aftershock Risk Predictor')
st.markdown("""
Predict the probability of aftershocks for recent earthquakes. High-risk events are flagged in **red** on the map.
""")

# Side bar controls
st.sidebar.header('Query')
region_name = st.sidebar.selectbox('Region', list(REGIONS.keys()), index=0)
bbox = REGIONS[region_name] # type: ignore

min_mag = st.sidebar.slider('Minimum magnitude', 0.0, max_mag, 3.0, 0.1) # min, max, starting, step

end_dt = datetime.now(timezone.utc)
start_dt = end_dt - timedelta(days=30)  # Default to 30 days for recent earthquakes
starttime = st.sidebar.text_input('Start (YYYY-MM-DD)', start_dt.strftime('%Y-%m-%d'))
endtime = st.sidebar.text_input('End (YYYY-MM-DD)', end_dt.strftime('%Y-%m-%d'))

limit = int(st.sidebar.number_input('Number of earthquakes', min_value=1, max_value=limit_num_eartquakes, value=500)) # type: ignore

auto_refresh = st.sidebar.checkbox('Auto-refresh', value=True)
refresh_seconds = st.sidebar.slider('Refresh (seconds)', 10,300,60) # type: ignore

# Initialize session state for tracking query parameters
# Initialize session state
if 'query_params' not in st.session_state:
    st.session_state.query_params = None

# Check if query parameters have changed
current_params = (region_name, starttime, endtime, min_mag, limit)
query_changed = st.session_state.query_params != current_params

# Determine if we should refresh data
# Refresh if: query changed, auto-refresh enabled, or first load
should_refresh = query_changed or auto_refresh or st.session_state.query_params is None

# Load data if needed
if should_refresh:
    data = load_data(starttime=starttime, endtime=endtime, min_mag=min_mag, bbox=bbox, limit=limit)
    st.session_state.data = data
    st.session_state.query_params = current_params
else:
    # Use cached data from session state
    if 'data' in st.session_state:
        data = st.session_state.data
    else:
        # First load
        data = load_data(starttime=starttime, endtime=endtime, min_mag=min_mag, bbox=bbox, limit=limit)
        st.session_state.data = data
        st.session_state.query_params = current_params

if data.empty:
    st.warning("No earthquakes found for the selected criteria. Try adjusting the date range or region.")
    st.stop()

if auto_refresh:
    st.caption(f'Auto-refresh enabled (every {refresh_seconds}s)')
else:
    st.caption('Auto-refresh disabled - data frozen. Change query parameters or enable auto-refresh to update.')

# Check if we need to reprocess (only if query changed or first time)
if query_changed or 'processed_data' not in st.session_state:
    with st.spinner("Generating predictions for selected query..."):
        # Process features
        df_feat = basic_time_feats(data)
        df_feat = add_seq_feat(df_feat)
        
        # Add aftershock label (for display purposes)
        df_labeled = add_aftershock_label(df_feat, T_hours=T_hours, R_km=R_km)
        df_labeled['region'] = region_name
        
        # Save to Hopsworks in background (silently, no user message)
        if USE_HOPSWORKS and not df_labeled.empty:
            try:
                save_features_to_hopsworks(df_labeled)
            except Exception:
                pass  # Fail silently - not critical for user experience
        
        # Load model and generate predictions
        # Use Streamlit cache to keep model and connection alive across reruns
        model = None
        if USE_HOPSWORKS:
            try:
                # Cache model loading to avoid re-login on every rerun
                if 'cached_model' not in st.session_state:
                    st.session_state.cached_model = load_model_from_hopsworks()
                model = st.session_state.cached_model
                
                df_feat["p_aftershock"] = predict_aftershock_proba(model, df_feat)
                df_labeled["p_aftershock"] = predict_aftershock_proba(model, df_labeled)
            except Exception as e:
                # Log error for debugging
                import traceback
                error_msg = str(e)
                error_trace = traceback.format_exc()
                # Store error in session state for debugging
                st.session_state.model_load_error = error_msg
                st.session_state.model_load_trace = error_trace
                # Log to console for debugging
                print(f"ERROR loading model: {error_msg}")
                print(f"Traceback: {error_trace}")
                # Don't show error to user, just continue without model
                pass  # Model not available - predictions will be None
        
        # Cache processed data
        st.session_state.processed_data = {
            'df_feat': df_feat,
            'df_labeled': df_labeled,
            'model': model
        }
        st.session_state.query_params = current_params
else:
    # Use cached processed data
    df_feat = st.session_state.processed_data['df_feat']
    df_labeled = st.session_state.processed_data['df_labeled']
    model = st.session_state.processed_data['model']

# Calculate metrics
mstats = mag_stats(df_feat)
dstats = depth_stats(df_feat)

if auto_refresh:
    st.caption(f'Auto-refresh every {refresh_seconds}s')



# ============================================================================
# EARTHQUAKE STATISTICS OVERVIEW
# ============================================================================
st.header('Earthquake Statistics Overview')

overview_cols = st.columns(4)
overview_cols[0].metric('Total Events', len(df_feat))
overview_cols[1].metric('Max Magnitude', f"{mstats['max']:.2f}" if mstats['max'] is not None else '-') # type: ignore
overview_cols[2].metric('Avg Depth (km)', f"{dstats['mean']:.1f}" if dstats['mean'] is not None else '-') # type: ignore
overview_cols[3].metric('Region', region_name)

# ============================================================================
# AFTERSHOCK PREDICTION RESULTS
# ============================================================================
if model is not None and 'p_aftershock' in df_feat.columns and df_feat['p_aftershock'].notna().any():
    st.header('Aftershock Prediction Results')
    
    # Calculate risk metrics
    pred_df = df_feat[df_feat['p_aftershock'].notna()].copy()
    pred_mean = pred_df['p_aftershock'].mean()
    pred_max = pred_df['p_aftershock'].max()
    pred_high_risk = (pred_df['p_aftershock'] > 0.5).sum()
    pred_very_high_risk = (pred_df['p_aftershock'] > 0.7).sum()
    
    # Show prediction metrics
    pred_cols = st.columns(4)
    pred_cols[0].metric('Avg Risk', f'{pred_mean:.1%}')
    pred_cols[1].metric('Max Risk', f'{pred_max:.1%}')
    pred_cols[2].metric('High Risk Events', f'{pred_high_risk}')
    pred_cols[3].metric('Very High Risk', f'{pred_very_high_risk}')
    
    # Show high-risk warnings
    if pred_high_risk > 0:
        st.warning(f"⚠️ **{pred_high_risk} earthquake(s) have high aftershock risk (probability > 50%)**")
        
        # Show details of high-risk events
        high_risk_events = pred_df[pred_df['p_aftershock'] > 0.5].sort_values('p_aftershock', ascending=False)
        with st.expander(f"View {pred_high_risk} High-Risk Event(s) Details", expanded=True):
            for idx, row in high_risk_events.iterrows():
                risk_color = "🔴" if row['p_aftershock'] > 0.7 else "🟠"
                st.markdown(
                    f"{risk_color} **{row.get('place', 'Unknown location')}** - "
                    f"Magnitude: {row.get('magnitude', 'N/A'):.1f}, "
                    f"Risk: **{row['p_aftershock']:.1%}** "
                    f"({row['time'].strftime('%Y-%m-%d %H:%M') if hasattr(row['time'], 'strftime') else row['time']})"
                )
    
    if pred_very_high_risk > 0:
        st.error(f"🚨 **{pred_very_high_risk} earthquake(s) have VERY HIGH aftershock risk (probability > 70%)** - Exercise extreme caution!")
elif model is None:
    # Only show error to user if model is not available
    st.info("ℹ️ Model not available. Train a model using: `python scripts/train_model.py`")



# ============================================================================
# VISUALIZATIONS
# ============================================================================


# Map visualization - full width at top
st.subheader('Earthquake Map')
if model is not None and 'p_aftershock' in df_feat.columns and df_feat['p_aftershock'].notna().any():
    st.caption("🔴 Red = High Risk | 🟠 Orange = Medium Risk | 🟢 Green = Low Risk")
deck = map_plot(df_feat)
if deck is None:
    st.info('No events found for this query')
else:
    st.plotly_chart(deck, use_container_width=True, height=500)

# Analysis visualizations in a grid below
st.subheader('Trends & Analysis')

# Create columns for different visualizations
if model is not None and 'p_aftershock' in df_feat.columns and df_feat['p_aftershock'].notna().any():
    # If we have predictions, show 4 visualizations in 2x2 grid
    col1, col2 = st.columns(2, gap='medium')
    
    with col1:
        st.plotly_chart(freq_plot(df=df_feat, freq='D'), use_container_width=True)
        st.plotly_chart(mag_hist(df=df_feat, bins=30), use_container_width=True)
    
    with col2:
        import plotly.express as px
        pred_df = df_feat[df_feat['p_aftershock'].notna()].copy()
        if not pred_df.empty:
            pred_df = pred_df.sort_values('time')
            fig_timeline = px.line(
                pred_df, 
                x='time', 
                y='p_aftershock',
                title='Risk Over Time',
                labels={'p_aftershock': 'Aftershock Probability', 'time': 'Time'}
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
            
            fig_dist = px.histogram(
                pred_df,
                x='p_aftershock',
                nbins=30,
                title='Risk Distribution',
                labels={'p_aftershock': 'Aftershock Probability', 'count': 'Number of Events'}
            )
            st.plotly_chart(fig_dist, use_container_width=True)
else:
    # If no predictions, show 2 visualizations side by side
    col1, col2 = st.columns(2, gap='medium')
    
    with col1:
        st.plotly_chart(freq_plot(df=df_feat, freq='D'), use_container_width=True)
    
    with col2:
        st.plotly_chart(mag_hist(df=df_feat, bins=30), use_container_width=True)


# ============================================================================
# DETAILED DATA TABLES
# ============================================================================
st.header('Detailed Data')

# Predictions table (if model is available)
if model is not None and 'p_aftershock' in df_labeled.columns and df_labeled['p_aftershock'].notna().any():
    st.subheader('Top Risk Events')
    pred_df = df_labeled[['time', 'place', 'magnitude', 'depth', 'p_aftershock']].copy()
    pred_df = pred_df[pred_df['p_aftershock'].notna()].sort_values('p_aftershock', ascending=False)
    pred_df['p_aftershock'] = pred_df['p_aftershock'].apply(lambda x: f'{x:.1%}')
    pred_df.columns = ['Time', 'Location', 'Magnitude', 'Depth (km)', 'Aftershock Risk']
    st.dataframe(pred_df.head(20), use_container_width=True, hide_index=True)
    st.caption(f"Showing top 20 events by aftershock risk. Total events analyzed: {len(pred_df)}")

# All events table
with st.expander("View All Events", expanded=False):
    display_df = df_feat[['time', 'place', 'magnitude', 'depth', 'latitude', 'longitude']].copy()
    if 'p_aftershock' in df_feat.columns:
        display_df['Aftershock Risk'] = df_feat['p_aftershock'].apply(
            lambda x: f'{x:.1%}' if pd.notna(x) else '-'
        )
    display_df = display_df.sort_values('time', ascending=False)
    display_df.columns = ['Time', 'Location', 'Magnitude', 'Depth (km)', 'Latitude', 'Longitude', 'Aftershock Risk'] if 'Aftershock Risk' in display_df.columns else ['Time', 'Location', 'Magnitude', 'Depth (km)', 'Latitude', 'Longitude']
    st.dataframe(display_df, use_container_width=True, hide_index=True)




    
