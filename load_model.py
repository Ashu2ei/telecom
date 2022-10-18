from pycaret.classification import load_model,save_model,predict_model
import pandas as pd
import spark
loaded_model = load_model('mymodel')
data =['avg_cqi', 'avg_intersite_distance', 'avg_pusch_sinr', 'avg_rsrp','avg_rsrq', 'avg_session_duration', 
'avg_uplink_interference','Carrier_Bandwidth', 'earfcn_dl', 'entity_cell_id', 'entity_geohash','feat_carrier_bw',
 'feat_cell_load', 'feat_cell_throughput','feat_channel_number', 'feat_cqi', 'feat_device_price','feat_interference', 
 'feat_intersite_distnace', 'feat_pusch_sinr','feat_rsrp', 'feat_rsrq', 'feat_ta','hour_of_day', 'ipthru_kbps', 
 'jio_avg_asp', 'num_of_records','PRBUtil_dl', 'qci_9_tp', 'session_duration', 'ta', 'feat_cell_capacity', 'feat_price_ordinal', 
 'feat_cn_bw_ordinal']

df = pd.DataFrame(data)
# df['coun'] = data
#predictions = predict_model(loaded_model, data=data)
#print(data)