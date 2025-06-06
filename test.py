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
MODEL_PATH ='model_vgg16.h5'
model = load_model(MODEL_PATH)

@app.route('/')
def index():
	return render_template('index3.html')


def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))

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
        preds = "Bacterial Blight - " + "Recommended Medicine: Streptomycin"
    elif preds == 1:
        preds = "Curl Virus - " + "Recommended Medicine: Saaf Fungicide"
    elif preds == 2:
        preds = "Fussarium Wilt - " + "Recommended Medicine: Redomil Gold Fungicide"
    else:
        preds = "Healthy"

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





@app.route('/cot_')
def cot_():
    return render_template('index2.html')
@app.route('/tom_')
def tom_():
    return render_template('tomato.html')
if __name__ == '__main__':
	app.run(debug=True)
