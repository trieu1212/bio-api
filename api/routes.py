from flask import Blueprint
from api.handler.userHandler import create_user
from api.handler.authHandler import verify_face, register_face 

auth_bp = Blueprint('auth_bp', __name__)
user_bp = Blueprint('user_bp', __name__)

user_bp.route('/create', methods=['POST'])(create_user)
auth_bp.route('/verify', methods=['POST'])(verify_face)
auth_bp.route('/register-face', methods=['POST'])(register_face)