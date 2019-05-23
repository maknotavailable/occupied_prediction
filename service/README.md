# Room Occupancy Prediction Service

## Setup
- Step 1: Install/start Docker on your computer
- Step 2: open your favorite command line interface
- Step 3: cd to ./service
- Step 4: enter -> docker build -t wattxdemo .
- step 5: enter -> docker run -d -p 5000:5000 wattxdemo:latest
- Step 6: open your browser and try the following link -> http://localhost:5000/?device_id=6&current_time=2016-07-30 23:59:59

## Notes
The containerized service only uses the classification model. Due to a lack of time, the predictions are based on the day and historical information about a certain meeting room - but only related to it's past availability for given days and time periods. 