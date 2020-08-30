from dropout import DropOut
from flask import Flask, abort, jsonify, request, render_template
import json
import os
from flask_cors import CORS, cross_origin
import io
app = Flask(__name__, static_url_path='',
            static_folder='static',
            template_folder='templates')
# refers to application_top
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
APP_STATIC = os.path.join(APP_ROOT, 'static')
cors = CORS(app, resources={r"/foo": {"origins": "http://localhost:4100"}})
app.config['CORS_HEADERS'] = 'Content-Type'
genDO = DropOut()


@app.route('/predict', methods=['POST', 'GET'])
@cross_origin()
def trace_bulk_number():
    json_id = request.get_json()
    paramDO = json_id['params']
    if len(paramDO) != 5 :
        data = genDO.predict(paramDO)
    else:
        data = {"error": "params length are not equal"}
    return jsonify(data)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
