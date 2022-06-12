# -*- coding: utf-8 -*-
# @Author : LuoXianan
# @File : app.py.py
# @Project: ML Deployment
# @CreateTime : 2022/5/12 22:26:46
import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle

# Create flask app
app = Flask(__name__)

# Load teh pickle model
model = pickle.load(open("model.pkl","rb"))  # 调用pickl加载model.pkl文件

# 定义主页
@app.route("/")
def Home():
    return render_template('index.html')

@app.route("/predict",methods=["POST"])   # 当我们的应用程序转到此页面时，预测，此时，predict方法被调用
def predict():
    floa_features = [float(x) for x in request.form.values()]
    features = [np.array(floa_features)]
    prediction = model.predict(features)

    return render_template('index.html', prediction_text ="The enterprise risk level is{}".format(prediction))

if __name__ == "__main__":
    app.run(debug=True)


