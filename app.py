from flask import Flask,render_template,request,jsonify
import json
import pickle
import numpy as np

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict",methods=["post"])
def predict_iris():
    data = request.form
    # return data
    with open("model.pkl","rb") as f:
        pfile = pickle.load(f)
    with open("asset.json","r") as f:
        jfile = json.load(f)
    ta = np.zeros(len(jfile["columns"]))
    ta[0] = int(data["SepalLengthCm"])
    ta[1] = int(data["SepalWidthCm"])
    ta[2] = int(data["PetalLengthCm"])
    ta[3] = int(data["PetalWidthCm"])
    # return jsonify({"Result":[test_array]})
    species = pfile.predict([ta])

    return jsonify({"message":{species}})

app.run(debug=True)