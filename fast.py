# 1. Library imports
import pandas as pd
from pycaret.regression import load_model, predict_model
from fastapi import FastAPI
import uvicorn

# 2. Create the app object
app = FastAPI()

#. Load trained Pipeline
#model = load_model('mymodel')


# 1. Library imports
import pandas as pd
from pycaret.regression import load_model, predict_model
from fastapi import FastAPI
import uvicorn

# 2. Create the app object
app = FastAPI()

#. Load trained Pipeline
model = load_model('mymodel')

# Define predict function
@app.post('/predict')
def predict(avg_cqi, avg_intersite_distance, avg_pusch_sinr, avg_rsrp, avg_rsrq, avg_session_duration,
    avg_uplink_interference, Carrier_Bandwidth, earfcn_dl, entity_cell_id, entity_geohash, feat_carrier_bw, 
    feat_cell_load, feat_cell_throughput, feat_channel_number, feat_cqi, feat_device_price, feat_interference, 
    feat_intersite_distnace, feat_pusch_sinr, feat_rsrp, feat_rsrq, feat_ta, hour_of_day, ipthru_kbps, jio_avg_asp, num_of_records, 
    PRBUtil_dl, qci_9_tp, session_duration, ta, feat_cell_capacity, feat_price_ordinal, feat_cn_bw_ordinal):
    data = pd.DataFrame([[avg_cqi, avg_intersite_distance, avg_pusch_sinr, avg_rsrp, avg_rsrq, avg_session_duration, avg_uplink_interference,
     Carrier_Bandwidth, earfcn_dl, entity_cell_id, entity_geohash, feat_carrier_bw, feat_cell_load, feat_cell_throughput,
      feat_channel_number, feat_cqi, feat_device_price, feat_interference, feat_intersite_distnace, feat_pusch_sinr,
       feat_rsrp, feat_rsrq, feat_ta, hour_of_day, ipthru_kbps, jio_avg_asp, num_of_records, PRBUtil_dl, qci_9_tp, 
       session_duration, ta, feat_cell_capacity, feat_price_ordinal, feat_cn_bw_ordinal]])
    data.columns = ['avg_cqi', 'avg_intersite_distance', 'avg_pusch_sinr', 'avg_rsrp','avg_rsrq', 'avg_session_duration', 
    'avg_uplink_interference','Carrier_Bandwidth', 'earfcn_dl', 'entity_cell_id', 'entity_geohash','feat_carrier_bw',
    'feat_cell_load', 'feat_cell_throughput','feat_channel_number', 'feat_cqi', 'feat_device_price','feat_interference', 
    'feat_intersite_distnace', 'feat_pusch_sinr','feat_rsrp', 'feat_rsrq', 'feat_ta','hour_of_day', 'ipthru_kbps', 
    'jio_avg_asp', 'num_of_records','PRBUtil_dl', 'qci_9_tp', 'session_duration', 'ta', 'feat_cell_capacity', 'feat_price_ordinal', 
    'feat_cn_bw_ordinal']

    predictions = predict_model(model, data=data) 
    return {'prediction': int(predictions['Label'][0])}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)