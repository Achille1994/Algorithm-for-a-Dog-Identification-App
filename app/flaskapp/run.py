# import packages
import keras
import tensorflow
import json
from flaskapp import app

from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, Conv2D, MaxPooling2D, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.preprocessing import image
from extract_bottleneck_features import *
from files.essential import *

import numpy as np
import cv2

from flask import render_template, request, Flask, redirect, url_for, flash 
from werkzeug.utils import secure_filename

import json
import os

UPLOAD_FOLDER = 'flaskapp/static/images/'
ALLOWED_EXTENSIONS = { 'png', 'jpg', 'jpeg', 'gif','tif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER 


# load dog names
with open('dog_names.json', 'r') as f:
    dog_names = json.load(f)


# master webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/master')
def master():
    
    # render web page 
    return render_template('master.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# web page that handles user file and displays model results
@app.route('/go', methods=['GET', 'POST'])
def go():
    # save user input file  
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file=request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # return redirect(url_for('Uploader_file', name=filename))
            predict_breed = dog_app(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            
    # This will render the go.html Please see that file. 
    return render_template('go.html', filename=filename,dog_names=dog_names,predict_breed=predict_breed)


#if __name__ == '__main__':
    #app.run(host='0.0.0.0', port=3000, debug=True)
    
