# Future improvements on the project can be implementing ETAS -> ML method
# DBSCAN
# Let's try logistic regression for now


# To run file use: streamlit run app.py

#import streamlit as st
#st.title('Earthquake Risk Predictor')
#st.write('Setup is working!')


'''
Missing tasks:
1) Implement feature scripts
2) Hopsworks connection
3) Implement the ML scripts
4) Deployment
'''




import streamlit as st
import pandas as pd
from datetime import datetime, timedelta, timezone
from src.viz import mag_hist, map_plot, freq_plot
from src.usgs_client import get_earthquakes
import time


# Loading of earthquakes
@st.cache_data(ttl=60, show_spinner=False)
def load_data(starttime:str, endtime:str, min_mag:float, bbox, limit:int = 10):
    df = get_earthquakes(starttime=starttime,endtime=endtime,min_magnitude=min_mag, bbox=bbox, limit=limit)
    return df



# Let's define some regions to test
REGIONS = {
    'Japan': (122,24,146,46),
    'Mexico': (-118,14,-86,-33),
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

min_mag = st.sidebar.slider('Minimum magnitude', 0.0, 8.0, 3.0, 0.1) # min, max, starting, step

days_back = st.sidebar.slider('Days back', 1, 30, 7) # We can change this slider later
end_dt = datetime.now(timezone.utc)
start_dt = end_dt - timedelta(days=days_back)
starttime = st.sidebar.text_input('Start (YYYY-MM-DD)', start_dt.strftime('%Y-%m-%d'))
endtime = st.sidebar.text_input('End (YYYY-MM-DD)', end_dt.strftime('%Y-%m-%d'))

limit = int(st.sidebar.number_input('Number of earthquakes', min_value=1, max_value=1000)) # type: ignore

auto_refresh = st.sidebar.checkbox('Auto-refresh', value=True)
refresh_seconds = st.sidebar.slider('Refresh (seconds)', 10,300,60) # type: ignore

#fetch = st.form_submit_button('Fetch Data')

#if fetch:
data = load_data(starttime=starttime, endtime=endtime, min_mag=min_mag, bbox=bbox, limit=limit)

if auto_refresh:
    st.caption(f'Auto-refresh every {refresh_seconds}s (cache TTL is 60s).')
    # Check this code later
    #st.autorefresh(interval=refres_seconds*1000, key='refresh')
    #time.sleep(refresh_seconds)
    #st.experimental_rerun()

# Summary
c1, c2, c3, c4 = st.columns(4)
c1.metric('Events', len(data))
c2.metric('Max magnitude', f"{data['magnitude'].max():.2f}" if len(data) else '-') # type: ignore
c3.metric('Avg depth (km)', f"{data['depth'].mean():.1f}" if len(data) else '-') # type: ignore
c4.metric('Region', region_name)

left, right = st.columns([1.2,1.0], gap='large')

with left:
    st.subheader('Map')
    deck = map_plot(data)
    if deck is None:
        st.info('No event found for this query')
    else:
        st.plotly_chart(deck, use_container_width=True)

with right:
    # We can change the parameters later
    st.plotly_chart(freq_plot(df=data,freq='D'), use_container_width=True)
    st.plotly_chart(mag_hist(df=data, bins=30), use_container_width=True)




    
