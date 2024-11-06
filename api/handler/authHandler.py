import pickle
import os
from flask import jsonify, request
import cv2
import numpy as np
import json
from deepface import DeepFace
from keras._tf_keras.keras.models import load_model
from config import Config
from api.service import userService
from model.face_recognize import detect_and_extract_embedding
from sklearn.metrics.pairwise import cosine_similarity


DATA_DIR = '/home/trieu/project/bio-python/data/faces'
EMBEDDINGS_PATH = '/home/trieu/project/bio-python/embeddings.pickle'
THRESHOLD = 0.7
label_map = {}

def verify_face():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image = request.files['image']
    image = cv2.imdecode(np.frombuffer(image.read(), np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({'error': 'Invalid image format'}), 400

    try:
        embedding_obj = DeepFace.represent(image, model_name="Facenet")
        if not embedding_obj:
            return jsonify({'error': 'No face detected'}), 400
        embedding = embedding_obj[0]["embedding"]
    except Exception as e:
        return jsonify({'error': f'Embedding extraction failed: {str(e)}'}), 500

    with open(EMBEDDINGS_PATH, 'rb') as f:
        known_embeddings = pickle.load(f)

    best_match = None
    highest_similarity = 0

    for user, embeddings_list in known_embeddings.items():
        for known_embedding in embeddings_list:
            similarity = cosine_similarity([embedding], [known_embedding])[0][0]
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = user

    if highest_similarity > THRESHOLD:
        return jsonify({
            'status': 'success',
            'user': best_match,
            'similarity': highest_similarity
        }), 200
    else:
        return jsonify({
            'status': 'failure',
            'message': 'User not recognized',
            'similarity': highest_similarity
        }), 400

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
    
    embeddings = []
    for i, image_file in enumerate(images):
        image_data = np.frombuffer(image_file.read(), np.uint8)
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        
        try:
            embedding_obj = DeepFace.represent(image, model_name="Facenet")
            if embedding_obj:
                embedding = embedding_obj[0]["embedding"]
                embeddings.append(embedding)
                image_path = os.path.join(user_dir, f'{i + 1}.jpg')
                cv2.imwrite(image_path, image)
        except Exception as e:
            return jsonify({'error': f'Embedding extraction failed for image {i + 1}: {str(e)}'}), 500

    if os.path.exists(EMBEDDINGS_PATH):
        with open(EMBEDDINGS_PATH, 'rb') as f:
            all_embeddings = pickle.load(f)
    else:
        all_embeddings = {}

    all_embeddings[f"{id}_{username}"] = embeddings
    with open(EMBEDDINGS_PATH, 'wb') as f:
        pickle.dump(all_embeddings, f)

    userService.update_label_user(f"{id}_{username}", id)
    
    return jsonify({'message': 'Face registered successfully'}), 200