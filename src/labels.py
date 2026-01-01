import pandas as pd
import numpy as np

from features import haversine_km_convert

def add_aftershock_label(df: pd.DataFrame, T_hours: float = 24.0, R_km: float = 100.0,
                         min_aftershock_mag: float | None = None):
    '''
    Adds binary label 'y_aftershock':
    y=1 if there exists a future event within the next T_hours AND R_km km
    '''
    out = df.copy()

    if out.empty:
        out['y_aftershock'] = []
        return out
    
    # At this point, the NaNs of Lat, Lon and time should already be dropped 
    # The df should already be sorted time-wise

    times_ns = out['time'].view('int64').to_numpy()
    lats = out['latitude'].astype(float).to_numpy()
    lons = out['longitude'].astype(float).to_numpy()
    mags = pd.to_numeric(out.get('magnitude', pd.Series([np.nan]*len(out))), errors='coerce').to_numpy()

    T_ns = int(T_hours * 3600 * 1e9)

    y = np.zeros(len(out),dtype=int)

    # Let's create a window T_hours into the future for events within R_km 
    for i in range(len(out)):
        t_end = times_ns[i] + T_ns

        # find end index for which time > t_end
        k = i + 1
        while k < len(out) and times_ns[k] <= t_end:
            k+=1
        
        if k == i + 1:
            continue # There are no event within the desired time window

        # Compute distances from event i to i+1,...,k
        cand_idx = slice(i+1,k)
        
        lat_i = pd.Series([lats[i]] * (k - (i + 1)))
        lon_i = pd.Series([lons[i]] * (k - (i + 1)))
        lat_j = pd.Series(lats[cand_idx])
        lon_j = pd.Series(lons[cand_idx])

        dists = haversine_km_convert(lat_i, lon_i, lat_j, lon_j).to_numpy()

        if min_aftershock_mag is not None:
            cand_mags = mags[cand_idx]
            acceptable_mag = np.nan_to_num(cand_mags, nan=-999.0) >= float(min_aftershock_mag)
            hit = np.any((dists<=R_km) & acceptable_mag)
        else:
            hit = np.any(dists<=R_km)
        y[i] = 1 if hit else 0
    out['y_aftershock'] = y
    return out


if __name__=='__main__':
    from usgs_client import get_earthquakes
    from datetime import datetime, timedelta
    from features import basic_time_feats, compute_freq_series, add_seq_feat

    days=30

    end = datetime.utcnow()
    start = end - timedelta(days=days)
    starttime = start.strftime('%Y-%m-%d')
    endtime = end.strftime('%Y-%m-%d')

    df = get_earthquakes(starttime=starttime,endtime=endtime,min_magnitude=None,limit=10)
    seq = basic_time_feats(df).head()
    seq = add_seq_feat(seq).head()

    out = add_aftershock_label(seq)
    print(out.head())




