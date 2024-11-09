import pickle
import os
from flask import jsonify, request
import cv2
import traceback
import numpy as np
import json
from deepface import DeepFace
from keras._tf_keras.keras.models import load_model
from config import Config
from api.service import userService
from model.face_recognize import detect_and_extract_embedding
from sklearn.metrics.pairwise import cosine_similarity
from model.face_recognize import train_embeddings, normalize_embedding


DATA_DIR = '/home/trieu/project/bio-python/data/faces'
EMBEDDINGS_PATH = '/home/trieu/project/bio-python/embeddings'
THRESHOLD = 0.3


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
        if isinstance(embedding, dict):
            return jsonify({'error': 'Embedding extraction failed: Unexpected format'}), 500
        embedding = normalize_embedding(np.array(embedding, dtype=float))
    except Exception as e:
        return jsonify({'error': f'Embedding extraction failed: {str(e)}'}), 500

    best_match = None
    highest_similarity = 0

    for filename in os.listdir(EMBEDDINGS_PATH):
        if filename.endswith(".pkl"):
            file_path = os.path.join(EMBEDDINGS_PATH, filename)
            try:
                with open(file_path, 'rb') as f:
                    known_embeddings = pickle.load(f)

                for known_embedding in known_embeddings:
                    if isinstance(known_embedding, dict):
                        print(f"Invalid format in {filename}, skipping this embedding: {known_embedding}")
                        continue

                    known_embedding = normalize_embedding(np.array(known_embedding, dtype=float))
                    similarity = float(cosine_similarity([embedding], [known_embedding])[0][0])
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        best_match = filename[:-4] 

            except Exception as e:
                print(f"Error loading or comparing embeddings from {filename}: {e}")
                traceback.print_exc()
                continue  

    print(f"Best match: {best_match}, similarity: {highest_similarity}")

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
    
    user_folder, embeddings = train_embeddings(id, username, images)
    print(f"Embeddings: {embeddings}")
    userService.update_label_user(user_folder, id)
    
    return jsonify({'message': 'Face registered successfully'}), 200