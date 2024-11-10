from flask import jsonify, request
from api.service import userService
from utils.hashPassword import hash_password

def create_user():
    username = request.form.get('username')
    password = request.form.get('password')
    email = request.form.get('email')
    role = request.form.get('role')

    if not username:
        return jsonify({'error': 'No username provided'}), 400
    if not password:
        return jsonify({'error': 'No password provided'}), 400
    if not email:
        return jsonify({'error': 'No email provided'}), 400
    if not role:
        return jsonify({'error': 'No role provided'}), 400

    user = userService.get_user_by_email(email)
    if user:
        return jsonify({'error': 'Email already exists'}), 400  
    
    hashed_password = hash_password(password).decode('utf-8')

    user_data = {
        'username': username,
        'password': hashed_password,
        'email': email,
        'role': role
    }

    res = userService.create_user(user_data)
    return jsonify(res), 200