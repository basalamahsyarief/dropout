import numpy as np
import pandas as pd
import time
import json
import os
import joblib
PATH = os.path.abspath(os.path.join(__file__, "./../"))


class DropOut:
    def __init__(self, dirpath=PATH+"/"):
        self.dirpath = dirpath
        self.model = None
        self.scaler = None
        self.load_model(dirpath+'model/checkpoint_dropout.model',
                        dirpath+'model/scaler.sav')

    def load_model(self, model_path, scaler_path):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def predict(self, input):
        input = np.array(input)
        input[1] = self.scaler.transform(input[1].reshape(1,-1))[0][0]
        result = self.model.predict(input.reshape(1,-1))
        proba = max(self.model.predict_proba(input.reshape(1,-1))[0])
        return {'result' : result[0], 'probability' : proba}
