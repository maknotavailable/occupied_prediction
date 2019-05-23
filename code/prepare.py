
import numpy as np
import pandas as pd
import datetime

def device_nr(d):
    """Turn Device ID into a numeric ID"""
    return int(d.split('_')[1])

def handle_duplicates(data, cols):
    """Detect and remove duplicates in the data"""
    for col in cols:
        print(f'[INFO] Device {col} length {len(data[data.device == col])}')
        if not len(data[data.device == col]) == len(data[data.device ==  col].drop_duplicates()):
            print(f'\t[ERROR] Device {col} contains {len(data[data.device == col]) - len(data[data.device ==  col].drop_duplicates())} duplicates.')
            data.drop_duplicates(inplace=True) 
    return data

def resample(data, window):
    """Resample the dataframe to a given sampling window, via aggregation"""
    data = data.groupby('device').apply(lambda x : x
                                        .set_index('timestamp', drop=False)
                                        .resample(window).agg(np.sum))
    data['device_activated'] = data['device_activated'].fillna(0)
    data['timestamp'] = pd.to_datetime(data.index.get_level_values('timestamp'))
    data['device'] = data.index.get_level_values('device').values
    return data

def get_occupied(data, column):
    """Convert resampled sums back to booleans"""
    return (data[column] != 0).astype(int)

def bin_time(dt):
    """Return binned time period

    -Options:
    0 = Night
    1 = Morning
    2 = Noon
    3 = Afternoon
    4 = Evening
    """
    dt = dt.time()
    if dt < datetime.time(hour=7):
        return 0
    elif dt < datetime.time(hour=12):
        return 1
    elif dt < datetime.time(hour=14):
        return 2
    elif dt < datetime.time(hour=16):
        return 3
    elif dt < datetime.time(hour=20):
        return 4
    else:
        return 0

def format_for_prophet(data):
    """Format data to match prophet input requirements"""
    return pd.DataFrame({'ds':data.index.get_level_values('timestamp'),'y':data.values})

def translate_score(score, thresh=1):
    """Simple conversion of forecast score to boolean"""
    score = round(score,0)
    if score < thresh:
        return 0
    else:
        return 1

def split(data, split=0.9):
    """Split data into a simple train and test set"""
    train = data[:int(split*(len(data)))].copy()
    valid = data[int(split*(len(data))):].copy()
    return train, valid