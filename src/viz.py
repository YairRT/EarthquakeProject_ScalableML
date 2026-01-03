import pandas as pd
import plotly.express as px

def map_plot (df: pd.DataFrame):
    if df.empty:
        return None
    
    # Use prediction probability for color if available, otherwise use magnitude
    has_predictions = 'p_aftershock' in df.columns and df['p_aftershock'].notna().any()
    
    if has_predictions:
        # Create risk categories for better visualization
        df_plot = df.copy()
        df_plot['risk_level'] = pd.cut(
            df_plot['p_aftershock'],
            bins=[0, 0.3, 0.5, 0.7, 1.0],
            labels=['Low (0-30%)', 'Medium (30-50%)', 'High (50-70%)', 'Very High (70-100%)']
        )
        
        # Color map: Low = green, Medium = yellow, High = orange, Very High = red
        color_map = {
            'Low (0-30%)': '#2ecc71',      # Green
            'Medium (30-50%)': '#f39c12',  # Orange/Yellow
            'High (50-70%)': '#e67e22',     # Dark Orange
            'Very High (70-100%)': '#e74c3c' # Red
        }
        
        df_plot['risk_color'] = df_plot['risk_level'].map(color_map)
        
        hover_data = {
            'depth': True, 
            'time': True, 
            'magnitude': True,
            'p_aftershock': ':.1%',
            'risk_level': True
        }
        
        fig = px.scatter_geo(
            df_plot,
            lat='latitude',
            lon='longitude',
            size='magnitude',
            color='risk_level',
            color_discrete_map=color_map,
            hover_name='place',
            hover_data=hover_data,
            projection='natural earth',
            labels={'risk_level': 'Aftershock Risk Level'},
            title='Earthquake Map - Color indicates Aftershock Risk'
        )
    else:
        # Fallback to magnitude if no predictions
        hover_data = {'depth': True, 'time': True, 'latitude': True, 'longitude': True}
        fig = px.scatter_geo(
            df,
            lat='latitude',
            lon='longitude',
            size='magnitude',
            color='magnitude',
            color_continuous_scale='Viridis',
            hover_name='place',
            hover_data=hover_data,
            projection='natural earth',
            labels={'magnitude': 'Magnitude'}
        )
    
    fig.update_layout(margin=dict(l=0,r=0,t=30,b=0))
    return fig

def freq_plot(df: pd.DataFrame, freq: str = 'D'):
    '''
    Docstring for freq_plot
    
    :param df: dataframe of earthquakes
    :param freq: we can set it to 'H' for hourly and 'D' for daily
    '''
    if df.empty:
        return px.line(pd.DataFrame({'time':[], 'count':[]}), x='time',y='count') # type: ignore
    tmp = df.copy()
    tmp['time'] = pd.to_datetime(tmp['time'])
    counts = tmp.set_index('time').resample(freq).size().reset_index(name='count')
    #print(counts)
    return px.line(counts, x='time',y='count')

def mag_hist(df: pd.DataFrame, bins: int = 30):
    if df.empty:
        return px.histogram(pd.DataFrame({'magnitude':[]}),x='magnitude')
    return px.histogram(df, x='magnitude',nbins=bins)

if __name__=='__main__':
    from usgs_client import get_earthquakes
    from datetime import datetime, timedelta
    days=30

    end = datetime.utcnow()
    start = end - timedelta(days=days)
    starttime = start.strftime('%Y-%m-%d')
    endtime = end.strftime('%Y-%m-%d')

    df = get_earthquakes(starttime=starttime,endtime=endtime,min_magnitude=5,limit=200)
    #print(df.head())

    # Test map_plot
    fig1 = map_plot(df)
    if fig1:
        fig1.show()
    
    # Test freq_plot
    fig2 = freq_plot(df,freq='D')
    fig2.show()

    # Test mag_hist
    fig3 = mag_hist(df)
    fig3.show()
