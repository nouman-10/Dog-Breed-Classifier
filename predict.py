import tensorflow as tf
import os
import numpy as np
from keras.preprocessing import image
from train import load_dataset, define_model
import cv2

def label_to_category_dict(path):
    '''Returns a dictionary that maps labels to categories'''
    categories = os.listdir('Data/dogImages/train/')
    label_to_cat = map(lambda x: (int(x.split('.')[0]) - 1, x.split('.')[1]), categories)
    label_to_cat = {label: category for label, category in label_to_cat}
    return label_to_cat

train_files, train_targets = load_dataset('Data/dogImages/train')
label_to_cat = label_to_category_dict(train_files)
Resnet50_model = define_model((1, 1, 2048), 133)

Resnet50_model.load_weights('saved_models/weights_best_Resnet50.hdf5')

def face_detector(img_path):
    '''Returns true if human face present in image'''
    face_cascade = cv2.CascadeClassifier('Data/haarcascades/haarcascade_frontalface_alt.xml')
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def extract_Resnet50(tensor):
    '''Returns the VGG16 features of the tensor'''
    return tf.keras.applications.resnet50.ResNet50(weights='imagenet', include_top=False).predict(tf.keras.applications.resnet50.preprocess_input(tensor))

def path_to_tensor(img_path):
    '''Converts the image in the given path to a tensor'''
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def predict_breed(img_path):
    '''Predicts the breed of the given image'''
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    bottleneck_feature = tf.keras.models.Sequential([
                            tf.keras.layers.GlobalAveragePooling2D(input_shape=bottleneck_feature.shape[1:])
                        ]).predict(bottleneck_feature).reshape(1, 1, 1, 2048)
    # obtain predicted vector
    predicted_vector = Resnet50_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return label_to_cat[np.argmax(predicted_vector)]


def dog_breed_classifier(image_path):
    '''
       Returns the breed of the dog in the image if present.
       If a human is present, predicts the most resembling dog breed
    '''
    is_face_detected = face_detector(image_path)
    dog_breed = predict_breed(image_path)
    return dog_breed, is_face_detected