from flask import Flask, send_from_directory, request, jsonify
import os
from flask_cors import CORS, cross_origin
from interface import *

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

app = Flask(__name__, static_folder="./dist", template_folder="./dist")

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serveWebPage(path):
    if path and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

@app.route('/winChance', methods = ['GET'])
@cross_origin()
def getBattleInfoFromForm ():
    dico = solveBattle (request.args.get("battleInfo"))

    return jsonify(dico)


if __name__ == '__main__':
    # change directory to containing directory
    abspath = os.path.abspath(__file__)
    dir_name = os.path.dirname(abspath)
    os.chdir(dir_name)
    # start webapp
    app.run(debug=True, host="0.0.0.0")