from flask import Flask, render_template, send_from_directory
import os

app = Flask(__name__, static_folder="./dist", template_folder="./dist")

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def prout(path):
    if path and os.path.exists(app.static_folder + '/' + path):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')




if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')