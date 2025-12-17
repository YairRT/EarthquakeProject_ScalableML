import requests
import pandas as pd
from datetime import datetime, timedelta

USGS_BASE_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"

def get_earthquakes(starttime, endtime, min_magnitude, bbox=None, limit=10):
    '''
    Get eathquakes events from USGS API through its query service 
    NOTE: more params can be added following the APIs documentation
    
    :param starttime: ISO DATE/TIME format
    :param endtime: ISO DATE/TIME format
    :param min_magnitude: Limit to events with a magnitude smaller than the specified maximum
    :param bbox: To select a specific region (tuple) -> {min_lon, min_lat, max_lon, max_lat}
    :param limit: how many event we want to get
    '''

    params = {
        'format': 'geojson', # always get this format
        'starttime': starttime,
        'endtime': endtime,
        'minmagnitude': min_magnitude,
        'limit':limit,
        'orderby': 'time',
    }

    if bbox != None:
        params.update(
            {'minlongitude': bbox[0],
             'minlatitude': bbox[1],
             'maxlongitude': bbox[2],
             'maxlatitude': bbox[3]}
            )
    
    response = requests.get(USGS_BASE_URL, params=params)
    response.raise_for_status()

    data = response.json()
    return parse_from_geojson(data)

def parse_from_geojson(data):
    '''
    Get the details from each earthquake from docstring
    
    :param data: dictionary with the information from usgs website
    '''
    
    records = []

    for feature in data['features']:
        #print('#########')
        #print(feature)
        properties = feature['properties']
        geometry = feature['geometry']

        # More interesting features can be obtained from each earthquake
        # Just added the initial ones

        record = {'time': datetime.utcfromtimestamp(properties['time']/1000),
                  'magnitude': properties['mag'],
                  'place': properties['place'],
                  'longitude': geometry['coordinates'][0],
                  'latitude': geometry['coordinates'][1],
                  'depth': geometry['coordinates'][2],
                  'event_id': feature['id']
                  }
        records.append(record)

    df = pd.DataFrame(records)
    df = df.sort_values('time').reset_index(drop=True)

    return df

if __name__ == '__main__':
    end = datetime.utcnow()
    start = end -  timedelta(days=30)
    starttime = start.strftime('%Y-%m-%d')
    endtime = end.strftime('%Y-%m-%d')

    #bbox = {-125, 32, -113, 42}

    data = get_earthquakes(starttime=starttime, endtime=endtime, min_magnitude=6,limit= 5)
    print(data)

