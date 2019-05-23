import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt
# Custom functions
import prepare as pr

def plot_single(data, col, idx='device', w=15, h=5, start=None, end=None):
    """Plot a given column for all available devices in the data"""
    for _id in data.index.get_level_values(idx).drop_duplicates():
        plt.figure(figsize=(w,h))
        _data = data[data.index.get_level_values(idx) == _id].reset_index(level = 0, drop=True)
        if start:
            plt.axvline(x=start, color='r', linestyle='--')
        if end:
            plt.axvline(x=end, color='r', linestyle='--')
        plt.title(_id)
        plt.plot(_data.index, _data[col])
        plt.show()

def seasonal_distribution(data, device_id, column):
    """Plot seasonal distributions from the time series data
    
    Plots the following trends:
    - Weekly
    - Daily
    - Hourly
    """
    _data = data[data.index.get_level_values('device') == device_id][column]
    _data = pr.format_for_prophet(_data)
    m = Prophet(yearly_seasonality=False).fit(_data)
    future = m.make_future_dataframe(periods=24, freq='H')
    fcst = m.predict(future)
    fig = m.plot_components(fcst)