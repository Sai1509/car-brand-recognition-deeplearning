#Future Imports: Ensures compatibility with Python 2 and 3 for division and print functions.
#Standard Libraries: sys, os, glob, re, and numpy for system operations, file handling, and numerical operations.
#Keras Libraries: For loading and preprocessing images and the model.
#Flask Libraries: For creating a web application.
#Werkzeug Utilities: For securely handling file uploads.


from __future__ import division,print_function
import sys
import sys
import os
import glob
import re
import numpy as np


from tensorflow.keras.applications.imagenet_utils import preprocess_input,decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from flask import Flask, redirect,url_for,request,render_template
from werkzeug.utils import secure_filename


#Flask application setup
app=Flask(__name__)

#Model Path: Specifies the file path of the saved model.
#Load Model: Loads the pre-trained Keras model from the specified file.
#Model loading
MODEL_PATH='model_resnet50.h5'
#Load Model
model=load_model(MODEL_PATH)


#Model Prediction Function

def model_predict(img_path, model):
    print("Loading image...")
    img = image.load_img(img_path, target_size=(224, 224))
    
    # Preprocessing the image
    x = image.img_to_array(img)
    x = x / 255
    x = np.expand_dims(x, axis=0)
    
    print("Predicting...")
    preds = model.predict(x)
    preds = np.argmax(preds, axis=1)
    
    if preds == 0:
        preds = 'The car is Audi'
    elif preds == 1:
        preds = 'The car is Lamborghini'
    else:
        preds = 'The car is Mercedes'
    
    print("Prediction complete")
    return preds

#Home Route
#Index Route: Defines the route for the homepage which renders the index.html template.

@app.route('/',methods=['GET'])

def index():
        return render_template('index.html')
    
#Predict Route

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        print("File received: ", f.filename)
        
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        print("File saved to: ", file_path)
        
        # Making predictions
        preds = model_predict(file_path, model)
        result = preds
        return result
    return None


#Run Flask App
if __name__=='__main__':
    app.run(debug=True)

    








