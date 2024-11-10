import os
from utils.jwt import gen_jwt_token
from utils.hashPassword import check_password
from flask import jsonify, request
from config import Config
from api.service import userService
from model.face_recognize import train_embeddings
from api.service.authService import login_face_biometric

EMBEDDINGS_PATH = Config.EMBEDDINGS_DIR
THRESHOLD = float(Config.THRESHOLD)

def verify_face_login_biometrics():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    image = request.files['image']
    
    best_match, highest_similarity = login_face_biometric(image)

    if highest_similarity >= THRESHOLD:
            user_id = best_match.split('_')[0]
            user = userService.get_user_by_id(user_id)
            token = gen_jwt_token(user)
            return jsonify({
                'status': 'success',
                'user': user,
                'token': token,
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
    if not images or len(images) < 3:
        return jsonify({'error': 'At least 3 images are required'}), 400
    
    user_folder, embeddings = train_embeddings(id, username, images)
    print(f"Embeddings: {embeddings}")
    userService.update_label_user(user_folder, id)
    
    return jsonify({'message': 'Face registered successfully'}), 200

def login():
    email = request.form.get('email')
    password = request.form.get('password')
    if not email:
        return jsonify({'error': 'No email provided'}), 400
    if not password:
        return jsonify({'error': 'No password provided'}), 400

    user = userService.get_user_by_email(email)
    if user is None:
        return jsonify({'error': 'User not found'}), 400
    
    if not check_password(password, user['password'].encode('utf-8')):
        return jsonify({'error': 'Invalid password'}), 400

    token = gen_jwt_token(user)
    return jsonify({
        'status': 'success',
        'user': user,
        'token': token
    }), 200