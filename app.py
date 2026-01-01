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
1) Implement feature scripts
2) Hopsworks connection
3) Implement the ML scripts
#  Solve for optional to do list
4) Deployment
'''




import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, timezone
from src.viz import mag_hist, map_plot, freq_plot
from src.usgs_client import get_earthquakes
from src.features import add_seq_feat, basic_time_feats, mag_stats, depth_stats
import time
from src.labels import add_aftershock_label

# Variables
T_hours = 24
R_km = 100
limit_num_eartquakes = 1000
max_mag = 8.0


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

st.set_page_config(page_title='Earthquake Dashboard', layout='wide')
st.title('USGS Earthquakes - Live + Historical Explorer')
data = pd.DataFrame()

# Side bar controls
#with st.sidebar.form('queary_form'):
st.sidebar.header('Query')
region_name = st.sidebar.selectbox('Region', list(REGIONS.keys()), index=0)
bbox = REGIONS[region_name] # type: ignore

min_mag = st.sidebar.slider('Minimum magnitude', 0.0, max_mag, 3.0, 0.1) # min, max, starting, step

#days_back = st.sidebar.slider('Days back', 1, 30, 7) # We can change this slider later
end_dt = datetime.now(timezone.utc)
start_dt = end_dt - timedelta(days=7)
starttime = st.sidebar.text_input('Start (YYYY-MM-DD)', start_dt.strftime('%Y-%m-%d'))
endtime = st.sidebar.text_input('End (YYYY-MM-DD)', end_dt.strftime('%Y-%m-%d'))

limit = int(st.sidebar.number_input('Number of earthquakes', min_value=1, max_value=limit_num_eartquakes)) # type: ignore

auto_refresh = st.sidebar.checkbox('Auto-refresh', value=True)
refresh_seconds = st.sidebar.slider('Refresh (seconds)', 10,300,60) # type: ignore

#fetch = st.form_submit_button('Fetch Data')
# Features table
#show_feature_table = st.sidebar.checkbox('Show feature table', value=True)

#if fetch:
data = load_data(starttime=starttime, endtime=endtime, min_mag=min_mag, bbox=bbox, limit=limit)

if auto_refresh:
    st.caption(f'Auto-refresh every {refresh_seconds}s (cache TTL is 60s).')
    # Check this code later
    #st.autorefresh(interval=refres_seconds*1000, key='refresh')
    #time.sleep(refresh_seconds)
    #st.experimental_rerun()

# Features
df_feat = basic_time_feats(data)
df_feat = add_seq_feat(df_feat)

# Some metrics
mstats = mag_stats(df_feat)
dstats = depth_stats(df_feat)

# Add aftershock label
df_labeled = add_aftershock_label(df_feat, T_hours=T_hours, R_km=R_km)


# Summary
c1, c2, c3, c4 = st.columns(4)
c1.metric('Events', len(df_feat))
c2.metric('Max magnitude', f"{mstats['max']:.2f}" if mstats['max'] is not None else '-') # type: ignore
c3.metric('Avg depth (km)', f"{dstats['mean']:.1f}" if dstats['mean'] is not None else '-') # type: ignore
c4.metric('Region', region_name)

# Add metrics of new features created
if len(df_feat):
    c5, c6, c7, c8 = st.columns(4)
    c5.metric('Avg time since prev (h)',
              f'{df_feat["time_since_prev_hours"].dropna().mean():.2f}'
              if df_feat["time_since_prev_hours"].notna().any() else '-')
    c6.metric('Avg dist to prev (km)',
              f'{df_feat["distance_to_prev_km"].dropna().mean():.1f}'
              if df_feat["distance_to_prev_km"].notna().any() else '-')
    c7.metric("Max rolling 6h count",
              f'{int(df_feat["rolling_count_6h"].max())}' if 'rolling_count_6h' in df_feat.columns else '-')
    c8.metric("Max rolling 24h count",
              f'{int(df_feat["rolling_count_24h"].max())}' if 'rolling_count_24h' in df_feat.columns else '-')



left, right = st.columns([1.2,1.0], gap='large')

with left:
    st.subheader('Map')
    deck = map_plot(df_feat)
    if deck is None:
        st.info('No event found for this query')
    else:
        st.plotly_chart(deck, width='stretch')

with right:
    # We can change the parameters later
    st.subheader('Trends')
    st.plotly_chart(freq_plot(df=df_feat,freq='D'), width='stretch')
    st.plotly_chart(mag_hist(df=df_feat, bins=30), width='stretch')


# Tables

st.subheader('Raw events')

if True:
    st.subheader('Labeled events: aftershock count')
    st.write(df_labeled['y_aftershock'].value_counts())
    st.subheader('Events + engineered features')
    st.dataframe(df_feat.sort_values('time', ascending=False), width='stretch') # type: ignore




    
