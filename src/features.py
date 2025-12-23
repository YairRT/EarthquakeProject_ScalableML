import pandas as pd 
import numpy as np

def basic_time_feats(df: pd.DataFrame):
    '''
    Docstring for basic_time_feats
    
    :param df: dataframe of earthquake data 
    :type df: dataframe type with time columns (dayOfWeek, hour, etc)
    '''
    out = df.copy()
    out['time'] = pd.to_datetime(out['time'], utc=True, errors='coerce')
    out = out.dropna(subset=['time']).sort_values('time').reset_index(drop=True)

    out['hour'] = out['time'].dt.hour
    out['dayofweek'] = out['time'].dt.dayofweek # Monday = 0
    return out

def compute_freq_series(df: pd.DataFrame, freq: str = 'D'):
    '''
    Docstring for compute_freq_series
    
    :param df: earthquake df
    :param freq: 'Hourly', 'D' daily
    :returns: total counts pd.Dataframe
    '''

    if df.empty:
        return pd.DataFrame({'time':[],'count':[]})
    
    tmp = df.copy()
    tmp['time'] = pd.to_datetime(tmp['time'], utc=True, errors='coerce')
    tmp = tmp.dropna(subset=['time']) 

    counts = tmp.set_index('time').resample(freq).size().reset_index(name='count')

    return counts

def add_seq_feat(df: pd.DataFrame):
    '''
    Docstring for add_seq_feat
    
    Adds simple per-event 'sequence' features that we'll use for logistic reg.
        - time since_prev_hours
        - distance_to_prev_km
        - rolling_count_6hr, rolling_count_24h (based on event times)
    '''
    out = df.copy()
    if out.empty:
        return out
    
    #out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce") # can be eliminated
    out = out.dropna(subset=['time', 'latitude','longitude']).sort_values('time').reset_index(drop=True)

    # Time since previous event (hours)
    dt = out['time'].diff().dt.total_seconds() / 3600.0 # type: ignore
    out['time_since_prev_hours'] = dt.fillna(np.nan)

    # distance to previous event (km) using haversine
    out['distance_to_prev_km'] = haversine_km_convert(
        out['latitude'].shift(1),
        out['longitude'].shift(1),
        out['latitude'],
        out['longitude']
    )

    # Rolling counts how many quakes happened in the last X hours before each event
    # A rolling window aligned to each event timestam
    out = out.set_index('time')
    out['rolling_count_6h'] = out['event_id'].rolling('6h').count().shift(1)
    out['rolling_count_24h'] = out['event_id'].rolling('24h').count().shift(1)
    out = out.reset_index()

    out['rolling_count_6h'] = out['rolling_count_6h'].fillna(0).astype(int)
    out['rolling_count_24h'] = out['rolling_count_24h'].fillna(0).astype(int)

    return out

def depth_stats(df: pd.DataFrame):
    '''
    Write some statistics regarding depth
    '''
    d = pd.to_numeric(df['depth'], errors='coerce').dropna()
    if d.empty:
        return {'mean': None, 'p50': None, 'p90': None, 'max': None}
    
    return {'mean': float(d.mean()), 'p50': float(d.quantile(0.50)), 'p90': float(d.quantile(0.90)), 'max': float(d.max())}

def mag_stats(df: pd.DataFrame):
    '''
    Write some statis regarding mag
    '''
    m = pd.to_numeric(df['magnitude'], errors='coerce').dropna()
    if m.empty:
        return {'mean': None, 'p50': None, 'p90': None, 'max': None}
    
    return {'mean': float(m.mean()), 'p50': float(m.quantile(0.50)), 'p90': float(m.quantile(0.90)), 'max': float(m.max())}


# Now let's create a function to calculate distance in Km in between two geospatial points on earth
# Haversine is a good option for this purpose

def haversine_km_convert(lat1, lon1, lat2, lon2):
    '''
    Vector of Haversine distances transformed into KM
    '''

    R = 6371.0 # Radius of Earth in KMm

    # Convert from degrees to rad
    lat1 = np.radians(pd.to_numeric(lat1, errors='coerce'))
    lon1 = np.radians(pd.to_numeric(lon1, errors='coerce'))
    lat2 = np.radians(pd.to_numeric(lat2, errors='coerce'))
    lon2 = np.radians(pd.to_numeric(lon2, errors='coerce'))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # This formula can also be used but for numerical stability is better the other one
    # a = (1 - np.cos(dlat) + np.cos(lat1)* np.cos(lat2)*(1-np.cos(dlon)))/2 
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a)) # to calculate for the great-circle distance
    return R * c

if __name__=='__main__':
    from usgs_client import get_earthquakes
    from datetime import datetime, timedelta
    days=30

    end = datetime.utcnow()
    start = end - timedelta(days=days)
    starttime = start.strftime('%Y-%m-%d')
    endtime = end.strftime('%Y-%m-%d')

    df = get_earthquakes(starttime=starttime,endtime=endtime,min_magnitude=5,limit=200)

    print('Basic time features:')
    seq = basic_time_feats(df).head()
    print(seq)

    print('\nFrequency series (daily)')
    print(compute_freq_series(df).head())

    print('\nSequence features:')
    seq = add_seq_feat(seq).head()
    print(seq)
    print(seq.columns)

    print('\nStatistics of depth')
    print(depth_stats(df))

    print('\nStatistics of magnitude')
    print(mag_stats(df))