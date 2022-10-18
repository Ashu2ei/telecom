import pickletools
import model_service
import pickle
import mlflow
from PIL import Image
import numpy as np
from keras.utils import image_utils
import requests
from mlflow.models.signature import infer_signature
import tensorflow as tf

print('FIRST')

class ModelFunction:

        def __init__(self):
          self.model_service_obj = model_service.MlflowModelService() 

        def store_model(self,pickle_file): #pickle file will be in string format
          infile = open(pickle_file,'rb+')
          model = pickle.load(infile)
          infile.close()
          self.model_service_obj.saveModel(model,"pycaret","ImageClassification","preprocess.py")

        def load_model(self):
           self.model = self.model_service_obj.loadModel("ImageClassification","1")
           return self.model

# class_labels=[
# 	"Plane",
# 	"Car",
# 	"Bird",
# 	"Cat",
# 	"Deer",
# 	"Dog",
# 	"Frog",
# 	"Horse",
# 	"Horse",
# 	"Boat",
# 	"Truck"
# ]

obj = ModelFunction()
obj.store_model("mymodel.pkl")
model = obj.load_model()