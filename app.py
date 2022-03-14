from flask import Flask, url_for, redirect, render_template, request;
import numpy as np;
import pandas as pd;
import joblib;

app = Flask(__name__)

@app.route("/", methods = ["POST", "GET"])
def index():
    if request.method == "POST":
        weight = request.form["weight"]
        age = request.form["age"]
        model = joblib.load("regr.pkl")
        res = model.predict(pd.DataFrame([[age, weight]]))[0]
        return render_template("index.html", result = res)
    else:
        return render_template("index.html")

if __name__ == '__main__':
    app.debug = True
    app.run()