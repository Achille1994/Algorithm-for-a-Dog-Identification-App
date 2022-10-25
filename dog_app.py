# import packages
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.preprocessing import image
from extract_bottleneck_features import *
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse
import os
import keras
import tensorflow
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense,Conv2D, MaxPooling2D, Dense
from tensorflow.keras.models import Sequential
import json

# Avoid tensorflow print on standard error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Configure the input data of model
parser = argparse.ArgumentParser(description='Short sample  AI app for dog identification')
parser.add_argument('--input', default='dog_human/bulldog.png', action="store", type = str, help='Image path')

arg_parser = parser.parse_args()
image_path = arg_parser.input

def create_model():
    # create model
    Resnet50_Model = Sequential()
    Resnet50_Model.add(GlobalAveragePooling2D(input_shape=(7, 7, 2048)))
    Resnet50_Model.add(Dropout(0.4))
    Resnet50_Model.add(Dense(133, activation='softmax'))

    #Compile the model.
    Resnet50_Model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return Resnet50_Model

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151)) 

def Resnet50_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_Resnet50(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = Resnet50_Model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    breed_dog=dog_names[np.argmax(predicted_vector)]
    # return probability
    ps=predicted_vector[0][np.argmax(predicted_vector)]
    return breed_dog,ps


def face_detector(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0

def dog_app(img_path):
    # dog detector
    dog_detect=dog_detector(img_path)
    
    # face detector
    face_detect=face_detector(img_path)
    
    # if a dog is detected in the image, return the predicted breed. 
    if dog_detect==True:
        dog_img_BGR=cv2.imread(img_path)
        dog_img_RGB= cv2.cvtColor(dog_img_BGR, cv2.COLOR_BGR2RGB)
        
        #return the predicted breed
        breed_dog,ps=Resnet50_predict_breed(img_path)
        plt.imshow(dog_img_RGB)
        plt.title(f"Your predicted dog breed is:\n\n{breed_dog}\n\nPredict class is : {ps:.2f} ")
        plt.axis('off')
        plt.show()
        
    #if a human is detected in the image, return the resembling dog breed.
    elif face_detect==True:
        face_img_BGR=cv2.imread(img_path)
        face_img_RGB= cv2.cvtColor(face_img_BGR, cv2.COLOR_BGR2RGB)
        # return the resembling dog breed
        breed_dog,ps=Resnet50_predict_breed(img_path)
        plt.imshow(face_img_RGB)
        plt.title(f"This photo looks like: \n {breed_dog}")
        plt.axis("off")
        plt.show()
       
    # if neither is detected in the image, provide output that indicates an error.    
    else:
        print("Error, please provide human face image or dog image")


if __name__== "__main__":
    with open('dog_names.json', 'r') as f:
        dog_names= json.load(f)

    # define ResNet50 model
    ResNet50_model = ResNet50(weights='imagenet')

    # extract pre-trained face detector
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

    # reload the model
    Resnet50_Model=create_model()
    Resnet50_Model.load_weights('saved_models/weights.best.Resnet50.hdf5')

    dog_app(image_path)
    

    
    
