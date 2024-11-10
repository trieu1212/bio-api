from flask import Blueprint
from api.handler.userHandler import create_user
from api.handler.authHandler import verify_face_login_biometrics, register_face, login
from api.middleware import jwt_middleware

auth_bp = Blueprint('auth_bp', __name__)
user_bp = Blueprint('user_bp', __name__)

user_bp.route('/create', methods=['POST'])(create_user)
auth_bp.route('/login-biometrics', methods=['POST'])(verify_face_login_biometrics)
auth_bp.route('/login', methods=['POST'])(login)

auth_bp.route('/register-face', methods=['POST'])(jwt_middleware(register_face))