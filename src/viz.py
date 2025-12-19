import pandas as pd
import plotly.express as px

def map_plot (df: pd.DataFrame):
    if df.empty:
        return None
    fig = px.scatter_geo(df,
                         lat='latitude',
                         lon='longitude',
                         size='magnitude',
                         color='magnitude',
                         hover_name='place',
                         hover_data={'depth':True, 'time':True,'latitude':True, 'longitude':True},
                         projection='natural earth',
                         )
    fig.update_layout(margin=dict(l=0,r=0,t=0,b=0))
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
