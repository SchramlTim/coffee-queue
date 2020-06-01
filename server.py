from flask import Flask, request, g
from flask import send_file
from flask import jsonify

app = Flask(__name__, instance_relative_config=True)

@app.route('/count', methods=['get'])
def predict():
    f = open("tmp_count.txt", "r")
    people_count = f.read()
    return jsonify(people_count)

@app.route('/', methods=['get'])
def root():
    print('lol')
    return send_file('index.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0')