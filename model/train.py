import json
import os
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from model.model import load_data, create_model
from keras._tf_keras.keras.callbacks import ReduceLROnPlateau

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '../data/faces')
BATCH_SIZE = 8
EPOCHS = 30
MODEL_PATH = 'model.keras'



def train_model(data_dir, epochs=10):
    images, labels, label_map = load_data(data_dir)
    print(f"Kích thước hình ảnh: {images.shape}, Số lượng nhãn: {len(labels)}")
    with open('label_map.json', 'w') as f:
        json.dump(label_map, f)

    num_classes = len(label_map)
    print(f"Số lớp: {num_classes}")

    model = create_model(num_classes)
    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    images = images / 255.0
    datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6)
    model.fit(datagen.flow(images, labels, batch_size=BATCH_SIZE), epochs=EPOCHS, callbacks=[reduce_lr])

    model.save('saved_models/model.keras')

    return label_map