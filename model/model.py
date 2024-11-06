import json
import os
import keras
import numpy as np
import cv2
import tensorflow as tf

from keras._tf_keras.keras.models import Sequential, Model  
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, Input
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.preprocessing.image import load_img, img_to_array
  
BATCH_SIZE = 32
EPOCHS = 20

def load_data(data_dir):
    images = []
    labels = []
    label_map = {}   
    label_counter = 0 

    for person_dir in os.listdir(data_dir):
        person_path = os.path.join(data_dir, person_dir)
        if os.path.isdir(person_path):
            if person_dir not in label_map:
                label_map[person_dir] = label_counter
                label_counter += 1
            for img_file in os.listdir(person_path):
                img = cv2.imread(os.path.join(person_path, img_file))
                img = cv2.resize(img, (224, 224)) 
                images.append(img)
                labels.append(label_map[person_dir])

    return np.array(images), np.array(labels), label_map

def create_model(num_classes):
    base_model = keras.applications.MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)

    predictions = Dense(num_classes, activation='softmax')(x) 

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False

    for layer in base_model.layers[-50:]:
        layer.trainable = True

    return model