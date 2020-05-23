import os
from flask import Flask, render_template, request
from flask import send_from_directory
from predict import dog_breed_classifier
#from keras.models import load_model
#from keras.preprocessing import image
#import numpy as np
#import tensorflow as tf

app = Flask(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))
# UPLOAD_FOLDER = dir_path + '/uploads'
# STATIC_FOLDER = dir_path + '/static'
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'

# home page
@app.route('/')
def home():
   return render_template('index.html')


# procesing uploaded file and predict it
@app.route('/upload', methods=['POST','GET'])
def upload_file():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        file = request.files['image']
        full_name = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(full_name)
        dog_breed, is_face_detected = dog_breed_classifier(full_name)
        is_face_detected = 'Yes' if is_face_detected else 'No'
    return render_template('predict.html', image_file_name = file.filename, label = dog_breed, is_face_detected=is_face_detected)


@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.debug = True
    app.run(debug=True)
    app.debug = True