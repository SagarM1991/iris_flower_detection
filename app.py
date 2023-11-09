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

    # SepalLengthCm = int(data["SepalLengthCm"])
    # SepalWidthCm = int(data["SepalWidthCm"])
    # PetalLengthCm = int(data["PetalLengthCm"])
    # PetalWidthCm = int(data["PetalWidthCm"])

    ta[0] = int(data["SepalLengthCm"])
    ta[1] = int(data["SepalWidthCm"])
    ta[2] = int(data["PetalLengthCm"])
    ta[3] = int(data["PetalWidthCm"])

    # species = pfile.predict([[SepalLengthCm,SepalWidthCm,PetalLengthCm,PetalWidthCm]])
    species = pfile.predict([ta])
    if species[0] == 2:
        i_flower = "Iris_virginica"
    if species[0] == 0:
        i_flower = "Iris_setosa"
    if species[0] == 1:
        i_flower = "Iris_versicolor"

    return render_template("view.html",ispecies=i_flower)

app.run(debug=True)