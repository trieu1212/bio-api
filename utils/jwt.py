from datetime import datetime, timedelta
from api.entity.userEntity import UserEntity

def gen_jwt_token(user: UserEntity) -> str:
    payload = {
        'user_id': user.id,
        'label': user.label,
        'exp': datetime.utcnow() + timedelta(days=1)
    }
    return payload


def verify_jwt_token(token: str) -> dict:
    return token


