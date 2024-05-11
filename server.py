from flask import Flask, render_template, send_from_directory, request, make_response, jsonify
import os
from flask_cors import CORS, cross_origin
import re
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
    server_response = make_response(str(win_chance), 200)
    server_response.mimetype = "text/plain"
    return server_response
    




if __name__ == '__main__':
    # change directory to containing directory
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    # start webapp
    app.run(debug=True, host="0.0.0.0")