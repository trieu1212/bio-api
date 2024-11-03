import os
import numpy as np
import cv2

from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout
from keras._tf_keras.keras.optimizers import Adam

DATA_DIR = 'data/faces'   
IMG_SIZE = (128, 128)     
BATCH_SIZE = 32
EPOCHS = 20

def load_data(data_dir):
    images = []
    labels = []
    label_map = {}   
    label_count = 0 

    for label_name in os.listdir(data_dir):
        user_dir = os.path.join(data_dir, label_name)
        if os.path.isdir(user_dir):
            if label_name not in label_map:
                label_map[label_name] = label_count
                label_count += 1
            label = label_map[label_name]

            for image_name in os.listdir(user_dir):
                image_path = os.path.join(user_dir, image_name)
                image = cv2.imread(image_path)
                if image is not None:
                    image = cv2.resize(image, IMG_SIZE)
                    images.append(image)
                    labels.append(label)

    return np.array(images), np.array(labels), label_map

def create_model(num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model