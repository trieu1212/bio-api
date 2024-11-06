import json
from math import e
import os
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.callbacks import EarlyStopping
from model.model import load_data, create_model
from keras._tf_keras.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '../data/faces')
BATCH_SIZE = 32
EPOCHS = 50
MODEL_PATH = 'model.keras'



def train_model(data_dir, epochs=10):
    images, labels, label_map = load_data(data_dir)
    print(f"Kích thước hình ảnh: {images.shape}, Số lượng nhãn: {len(labels)}")
    with open('label_map.json', 'w') as f:
        json.dump(label_map, f)

    num_classes = len(label_map)
    print(f"Số lớp: {num_classes}")

    model = create_model(num_classes)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    images = images / 255.0

    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    datagen = ImageDataGenerator(
        rotation_range=45,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        channel_shift_range=20,
        fill_mode='nearest'
    )
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6) 
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    for layer in model.layers[:-1]:
        layer.trainable = False

    model.fit(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
              validation_data=(X_val, y_val),
              epochs=5, callbacks=[reduce_lr])
    
    for layer in model.layers:
        layer.trainable = True

    model.compile(optimizer=Adam(learning_rate=1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    model.fit(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
              validation_data=(X_val, y_val),
              epochs=epochs-5, 
              callbacks=[reduce_lr, early_stopping])

    model.save('saved_models/model.keras')

    return label_map