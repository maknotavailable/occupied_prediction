from flask import request, Flask, jsonify
import pandas as pd
import datetime
import pickle

# Initialize flask
app = Flask(__name__)

# Load model dependencies
with open('classify.ml', 'rb') as fn:
    m_classify = pickle.load(fn)

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

def create_features_24(data, device_id):
    """Pipeline for the feature engineering step for scoring"""

    _data = pd.DataFrame({'time':data,'device':[device_id for i in range(24)]})
    _data['weekday'] = _data.time.apply(lambda x: int(x.strftime('%w')))
    _data['timeperiod'] = _data.time.apply(lambda x: bin_time(x)).values
    _data.set_index('time', inplace=True)
    return _data

def classify(data):
    """Classification step based on environmental factors"""

    return m_classify.predict(data)


@app.route('/') 
def server(): 
    """Web Service to predict if a meeting room will be occupied"""

    current_time = request.args.get('current_time')
    device_id = request.args.get('device_id')

    # Default values for blank request
    if not current_time:
        current_time = pd.Timestamp.now(tz='Europe/Berlin')
    if not device_id:
        device_id = 3

    # Prepare data
    next_24_hours = pd.date_range(current_time, periods=24, freq='H').ceil('H')
    data24 = create_features_24(next_24_hours, int(device_id))
    # Score data
    res = classify(data24)
    # Prepare output
    predictions = pd.DataFrame({'time':next_24_hours.astype(str),
                            'device':['device_'+str(device_id) for i in range(24)],
                            'activation_predicted':res.astype(int)})

    return predictions.to_json(orient='records')


if __name__ == '__main__':
    app.run(host='0.0.0.0') 