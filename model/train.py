import os
import numpy as np
import json
from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.models import load_model
from model.model import load_data, create_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '../data/faces')
BATCH_SIZE = 32
EPOCHS = 20
MODEL_PATH = 'model.h5'

def load_trained_model(model_path, num_classes):
    try:
        model = load_model(model_path)
        print(f"Tải mô hình từ '{model_path}' thành công.")
    except:
        print("Không thể tải mô hình. Khởi tạo mô hình mới.")
        model = create_model(num_classes = num_classes)
    return model


images, labels, label_map = load_data(DATA_DIR)

with open('label_map.json', 'w') as f:
    json.dump(label_map, f)

images = images / 255.0
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)


num_classes = len(label_map)
model = load_trained_model(MODEL_PATH, num_classes=num_classes)
print(f"Number of classes: {num_classes}")

unique_labels = np.unique(labels)
print(f"Unique labels in dataset: {unique_labels}")
if np.any(unique_labels >= num_classes):
    raise ValueError(f"Some labels are outside the valid range: {unique_labels}")

model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, epochs = EPOCHS, batch_size = BATCH_SIZE, validation_data=(x_test, y_test))

model.save('model.h5')
