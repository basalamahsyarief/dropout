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
        self.model_cum = joblib.load(dirpath+'model/checkpoint_dropout.sav')
        self.model_single = joblib.load(dirpath+'model/checkpoint_dropout_per_semester')
        self.scaler = joblib.load(dirpath+'model/scaler.sav')

    def predict_cum(self, input):
        input = np.array(input)
        input[1] = self.scaler.transform(input[1].reshape(1,-1))[0][0]
        print(input)
        result = self.model_cum.predict(input.reshape(1,-1))
        dic = {0 : 'Tidak Tepat Waktu', 1 : 'Tepat Waktu'}
        proba = max(self.model_cum.predict_proba(input.reshape(1,-1))[0])
        return {'result' : dic[result[0]], 'probability' : proba}

    def predict_single(self, input):
        input = np.array(input)
        print(input)
        result = self.model_single.predict(input.reshape(1,-1))
        dic = {0 : 'Tidak Tepat Waktu', 1 : 'Tepat Waktu'}
        proba = max(self.model_single.predict_proba(input.reshape(1,-1))[0])
        return {'result' : dic[result[0]], 'probability' : proba}
