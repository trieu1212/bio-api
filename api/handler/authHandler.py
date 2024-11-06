from cProfile import label
import os
from flask import jsonify, request
import cv2
import numpy as np
import json
from keras._tf_keras.keras.models import load_model
from config import Config
from api.service import userService
from model.train import DATA_DIR, train_model

DATA_DIR = '/home/trieu/project/bio-python/data/faces'
model = load_model('saved_models/model.keras')

label_map = {}

def verify_face():
    model = load_model('saved_models/model.keras')
    with open('label_map.json', 'r') as f:
        label_map = json.load(f)

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image = request.files['image']
    image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({'error': 'Invalid image format'}), 400

    image = cv2.resize(image, (224, 224))
    image = image / 255.0  
    image = np.expand_dims(image, axis=0) 

    pred = model.predict(image)
    pred_label = np.argmax(pred[0])
    confidence = pred[0][pred_label]

    if float(confidence) > float(Config.THRESHOLD):
        label = [user for user, label in label_map.items() if label == pred_label][0]
        user = userService.get_user_by_label(label)
        user_dir = os.path.join(Config.FACES_DIR, f"{label}")
        os.makedirs(user_dir, exist_ok=True)

        existing_files = [f for f in os.listdir(user_dir) if f.endswith('.jpg')]
        if existing_files:
            latest_file = max([int(f.split('.')[0]) for f in existing_files])
            next_file_number = latest_file + 1
        else:
            next_file_number = 1

        image_copy = image.copy()
        if image_copy is None:
            print("Image copy is invalid.")

        image_path = os.path.join(user_dir, f'{next_file_number}.jpg')
        cv2.imwrite(image_path, image_copy) 

        print(f"Đã lưu ảnh mới vào: {image_path}")
        # train_model(DATA_DIR, epochs=50)

    probabilities = {}
    for i, prob in enumerate(pred[0]):
        for key, value in label_map.items():
            if int(value) == i:
              probabilities[key.split('_')[1]] = float(prob)
              break 

    print("Xác suất cho từng lớp:")
    for user_name, probability in probabilities.items():
        print(f"  {user_name}: {probability:.4f}")

    result = {
        'status': 'success' if user else 'failure',
        'username': user.get('username') if user else None,
        'user_id': str(user.get('_id')) if user else None,
        'label': label,
        'confidence': float(confidence),
    }
    
    if not user:
        if float(confidence) <= float(Config.THRESHOLD):
            result['error'] = 'Confidence level too low'
        else:
            result['error'] = 'User not found'
    
    return jsonify(result), 200

def register_face():
    id = request.form['id']
    username = request.form['username']
    if not id:
        return jsonify({'error': 'No id provided'}), 400
    
    user = userService.get_user_by_id(id)
    if user is None:
        return jsonify({'error': 'User not found'}), 400
    
    user_dir = os.path.join(Config.FACES_DIR, f"{id}_{username}")
    os.makedirs(user_dir, exist_ok=True)

    images = request.files.getlist('images')
    if not images or len(images) < 10:
        return jsonify({'error': 'At least 10 images are required'}), 400
    
    for i, image_file in enumerate(images):
        image_data = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (224, 224))
        
        image_path = os.path.join(user_dir, f'{i + 1}.jpg')
        print(f"Saving image to: {image_path}")
        cv2.imwrite(image_path, image)

    label_map = train_model(DATA_DIR, epochs=50)
    user_folder_name = f"{id}_{username}"
    label = label_map.get(user_folder_name)

    if label is not None:
        userService.update_label_user(user_folder_name, id)
    else:
        return jsonify({'error': 'Error registering face'}), 400

    return jsonify({'message': 'Face registered successfully'}), 200