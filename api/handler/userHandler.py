from flask import jsonify, request
from api.service import userService

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

    user_data = {
        'username': username,
        'password': password,
        'email': email,
        'role': role
    }

    res = userService.create_user(user_data)
    return jsonify(res), 200