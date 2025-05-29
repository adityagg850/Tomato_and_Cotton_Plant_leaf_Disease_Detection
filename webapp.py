from flask import Flask,render_template,url_for,request,jsonify
import numpy as np
import pandas as pd
import joblib,pickle
from werkzeug.utils import secure_filename
import tempfile
import sys
import os
import glob
import re
import joblib
import numpy
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
app = Flask(__name__)

app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPEG", "JPG", "PNG"]

# Model saved with Keras model.save()
MODEL_PATH ='tomato_disease48.h5'
MODEL_PATH1 ='inceptionv3.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model1 = load_model(MODEL_PATH1)
@app.route('/')
def index():
	return render_template('index.html')


def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(48, 48))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x = x / 255
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    # x = preprocess_input(x)

    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    if preds == 0:
        preds = "Bacterial Spot - " + "Recommended Medicine: Streptomycin"
    elif preds == 1:
        preds = "Early Blight - " + "Recommended Medicine: Saaf Fungicide"
    elif preds == 2:
        preds = "Late Blight - " + "Recommended Medicine: Redomil Gold Fungicide"
    elif preds == 3:
        preds = "Leaf Mold - " + "Recommended Medicine: Berrysun VC 100"
    elif preds == 4:
        preds = "Septoria Leaf Spot - " + "Recommended Medicine: Amistar Top Fungicide"
    elif preds == 5:
        preds = "Spider Mites Two Spotted Spider Mite - " + "Recommended Medicine: Alpha Bio"
    elif preds == 6:
        preds = "Target Spot - " + "Recommended Medicine: GeoLife No Virus"
    elif preds == 7:
        preds = "Tomato Yellow Leaf Curl Virus - " + "Recommended Medicine: Azadirachtin "
    elif preds == 8:
        preds = "Healthy"
    else:
        preds = "Mosaic Virus - " + "Recommended Medicine: Kavach Fungicide"

    return preds


def model_predict1(img_path, model1):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x = x / 255
    x = np.expand_dims(x, axis=0)

    preds = model1.predict(x)
    preds = np.argmax(preds, axis=1)
    if preds == 0:
        preds = "The leaf is a diseased cotton leaf"
    elif preds == 1:
        preds = "The leaf is a diseased cotton plant"
    elif preds == 2:
        preds = "The leaf is a fresh cotton leaf"
    else:
        preds = "The leaf is a fresh cotton plant"

    return preds





@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)
        result=preds
        return result
    return None



@app.route('/predict1', methods=['GET', 'POST'])
def upload1():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict1(file_path, model1)
        result=preds
        return result
    return None


@app.route('/cot_')
def cot_():
    return render_template('index2.html')
@app.route('/tom_')
def tom_():
    return render_template('tomato.html')
if __name__ == '__main__':
	app.run(debug=True)
