from cProfile import label
import os
import subprocess
from flask import jsonify, request
import cv2
import numpy as np
import model
import json
from utils import preprocessingImageApi
from keras._tf_keras.keras.models import load_model
from config import Config
from api.service import userService

model = load_model('model.h5')

label_map = {}
with open('label_map.json', 'r') as f:
    label_map = json.load(f)

def verify_face():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image = request.files['image']
    image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)

    processed_image = preprocessingImageApi.preprocess_image(image)
    predict_image = model.predict(processed_image)

    predicted_probability = np.max(predict_image[0])  
    print(predicted_probability)

    if predicted_probability > Config.THRESHOLD:
        predicted_class = np.argmax(predict_image[0])  
        label_name = [k for k, v in label_map.items() if v == predicted_class][0]
        return jsonify({'match': True, 'class': int(predicted_class), 'name': label_name}), 200  
    else:
        return jsonify({'match': False}), 200

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

        if image is None:
            return jsonify({'error': 'Invalid image'}), 400
        
        preprocessed_image = preprocessingImageApi.preprocess_image(image)  
        cv2.imwrite(os.path.join(user_dir, f'{i + 1}.jpg'), preprocessed_image)

    for _ in range(5):
        try:
            subprocess.run(['python', '-m', 'model.train.py'], check=True)
        except subprocess.CalledProcessError as e:
            return jsonify({'error': f'Training failed: {str(e)}'}), 500
    
    return jsonify({'message': 'Face registered successfully'}), 200