from dropout import DropOut
from flask import Flask, abort, jsonify, request, render_template
import json
import os
from flask_cors import CORS, cross_origin
import io
import joblib
import numpy as np
app = Flask(__name__, static_url_path='',
            static_folder='static',
            template_folder='templates')
# refers to application_top
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
APP_STATIC = os.path.join(APP_ROOT, 'static')
cors = CORS(app, resources={r"/foo": {"origins": "http://localhost:4100"}})
app.config['CORS_HEADERS'] = 'Content-Type'
genDO = DropOut()
PATH = os.path.abspath(os.path.join(__file__, "./../"))
dirpath = PATH+"/"
scaler = joblib.load(dirpath+'model/scaler.sav')
scaler_ipk = joblib.load(dirpath+'model/scaler_ipk.sav')
scaler_ips = joblib.load(dirpath+'model/scaler_ips.sav')
scaler_semester = joblib.load(dirpath+'model/scaler_semester.sav')
scaler_usia = joblib.load(dirpath+'model/scaler_usia.sav')


@app.route('/predict_cum', methods=['POST', 'GET'])
@cross_origin()
def predict_cum():
    dict_jkel = {'L': 0, 'P': 1}
    dict_kawin = {'Belum Kawin': 0, 'Janda / Duda': 1, 'Suami/Istri Meninggal': 2, 'Sudah Kawin': 3}
    json_id = request.get_json()
    umur = json_id['umur']
    jkel = dict_jkel[json_id['jkel']]
    ips1 = json_id['ips1']
    ips2 = json_id['ips2']
    ips3 = json_id['ips3']
    ips4 = json_id['ips4']
    ips5 = json_id['ips5']
    data = genDO.predict_cum([jkel, umur, ips1, ips2, ips3, ips4, ips5])
    return jsonify(data)

@app.route('/predict_single', methods=['POST', 'GET'])
@cross_origin()
def predict_single():
    dict_jkel = {'L': 0, 'P': 1}
    dict_masuk = {'BARU': 0, 'PINDAHAN': 1}
    dict_jenjang = {'D3': 0, 'PR': 1, 'S1': 2, 'S2': 3}
    dict_kawin = {'Belum Kawin': 0, 'Janda / Duda': 1, 'Suami/Istri Meninggal': 2, 'Sudah Kawin': 3}
    json_id = request.get_json()
    umur = json_id['umur']
    jkel = dict_jkel[json_id['jkel']]
    masuk = dict_masuk[json_id['masuk']]
    jenjang = dict_jenjang[json_id['jenjang']]
    kawin = dict_kawin[json_id['kawin']]
    semester = scaler_semester.transform(np.array(json_id['semester']).reshape(1,-1))[0][0]
    umur = scaler_usia.transform(np.array(json_id['umur']).reshape(1,-1))[0][0]
    ipk = scaler_ipk.transform(np.array(json_id['ipk']).reshape(1,-1))[0][0]
    ips = scaler_ips.transform(np.array(json_id['ips']).reshape(1,-1))[0][0]
    is_bekerja = json_id['is_bekerja']
    data = genDO.predict_single([semester, jkel, masuk, jenjang, umur, is_bekerja, kawin, ips, ipk])
    return jsonify(data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
