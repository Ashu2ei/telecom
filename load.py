import mlflow.pyfunc
from pycaret.clustering import predict_model
import os
from pycaret.clustering import load_model
model_name = "mymodel"
artifacts = {'pycaret_model_path': 'mymodel.pkl'}
#mlflow.set_tracking_uri('http://127.0.0.1:5000')
class PyCaretModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        model_path = context.artifacts['pycaret_model_path']
        self.model = load_model(os.path.splitext(model_path)[0])
        model_path = context.artifacts['pycaret_model_path']
        self.model = load_model(os.path.splitext(model_path)[0])
    def predict(self, data):
        # logged_model = \
        # '/Users/ashutosh20.mishra/Desktop/image_classification/Image-Recognition-CIFAR-10/telecom/mlruns/0/5ca525be57c84a0ca153a20b4d1cb4e9'
        logged_model = mlflow.pyfunc.load_model(logged_model)
        return predict_model(data)


with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(artifact_path=model_name,
                            python_model=PyCaretModel(),
                            artifacts=artifacts)

obj = PyCaretModel()
obj.predict(['avg_cqi', 'avg_intersite_distance', 'avg_pusch_sinr', 'avg_rsrp','avg_rsrq', 'avg_session_duration', 
'avg_uplink_interference',        'Carrier_Bandwidth', 'earfcn_dl', 'entity_cell_id', 'entity_geohash','feat_carrier_bw',
 'feat_cell_load', 'feat_cell_throughput',        'feat_channel_number', 'feat_cqi', 'feat_device_price',        
 'feat_interference', 'feat_intersite_distnace', 'feat_pusch_sinr','feat_rsrp', 'feat_rsrq', 'feat_ta', 'good_download_throughput',
'hour_of_day', 'ipthru_kbps', 'jio_avg_asp', 'num_of_records',        'PRBUtil_dl', 'qci_9_tp', 'session_duration', 'ta', 
  'feat_cell_capacity', 'feat_price_ordinal', 'feat_cn_bw_ordinal'])
