"""
Main orchestration for predicting the occupancy for meeting rooms in the 
upcoming 24 hours.

This file can be run as follows;
    score.py <current time> <input file name> <output file name>

Current time is the current hour and input file is all measured values from
the activation detector in each room for the past few hours.
"""

import sys
import pandas as pd
from fbprophet import Prophet
import pickle
# Custom functions
import prepare as pr

# Load model dependencies
with open('../assets/classify.ml', 'rb') as fn:
    m_classify = pickle.load(fn)

def create_features_24(data, device_id):
    """Pipeline for the feature engineering step for scoring"""

    _data = pd.DataFrame({'time':data,'device':[device_id for i in range(24)]})
    _data['weekday'] = _data.time.apply(lambda x: int(x.strftime('%w')))
    _data['timeperiod'] = _data.time.apply(lambda x: pr.bin_time(x)).values
    _data.set_index('time', inplace=True)
    return _data

def prepare_data(data):
    """Pipeline to format and prepare data for scoring"""

    _data = data.copy()
    _data['timestamp'] = pd.to_datetime(_data['time'])
    _data['device'] = _data['device'].apply(pr.device_nr)
    _data = pr.handle_duplicates(_data, _data.device.drop_duplicates().values)
    _data = pr.resample(_data, window='1H')
    _data['occupied'] = pr.get_occupied(_data, 'device_activated')

    return _data

def forecast(data, f_target='occupied'):
    """Forecasting step based purely on historical activation values"""

    _data = pr.format_for_prophet(data[f_target])
    m = Prophet(yearly_seasonality=False).fit(_data)
    future = m.make_future_dataframe(periods=48, freq='H')
    future['floor'] = 0
    fcst = m.predict(future)
    future['yhat'] = fcst['yhat']
    future = future[future['ds'] > start_time].tail(24)
    return future['yhat'].apply(lambda x: pr.translate_score(x)).values
    
def classify(data):
    """Classification step based on environmental factors"""

    return m_classify.predict(data)
    
def ensemble(data, data24, f_target='occupied'):
    """Ensemble the forecasting and classification engines"""
    res_forecast = forecast(data, f_target)
    res_classify = classify(data24)
    try:
        res = (res_forecast*0.4) + (res_classify*0.6)
    except Exception as e:
        print('[INFO] Not enough close by historical data points for forecasting.')
        res = res_classify
    return [round(value) for value in res]


def predict_future_activation(current_time, data):
    """This function predicts future hourly activation given previous sensings.
    
    INPUT
        - current_time (datetime object)    : example '2016-08-31 23:59:59'
        - data (pandas dataframe)           : containing historical sensor data
                                                and timestamps

    OUTPUT
        - predictions (pandas dataframe)    : timestamp, device, 
                                                predicted activation
    """
    global start_time
    
    # Prepare scoring data
    _data = prepare_data(data.copy())

    # Prepare result data
    next_24_hours = pd.date_range(current_time, periods=24, freq='H').ceil('H')
    predictions = pd.DataFrame([])
    start_time = current_time

    # Score per device
    for _device in data.device.unique():
        _device_id = int(_device.split('_')[1])
        _df = _data[_data.index.get_level_values('device') == _device_id].copy()
        _df24 = create_features_24(next_24_hours, _device_id)
        ## Ensemble
        _res = ensemble(_df, _df24)
        ## Format output
        _temp = pd.DataFrame({'time':next_24_hours,
                            'device':[_device for i in range(24)],
                            'activation_predicted':_res})
        if predictions.empty:
            predictions = _temp.copy()
        else: 
            predictions = predictions.append(_temp, ignore_index=True)
    predictions.set_index('time', inplace=True)
    predictions['activation_predicted'] = predictions['activation_predicted'].astype(int)

    return predictions

if __name__ == '__main__':
    current_time, in_file, out_file = sys.argv[1:]
    previous_readings = pd.read_csv(in_file)
    result = predict_future_activation(current_time, previous_readings)
    result.to_csv(out_file)