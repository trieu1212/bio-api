from flask import jsonify, request
from api.entity import userEntity
from api.entity.userEntity import UserEntity


def create_user(user_data):
    new_user = UserEntity(
        username=user_data["username"],
        password=user_data["password"],
        email=user_data["email"],
        role=user_data["role"]
    )
    new_user.save()
    print(new_user.to_dictionary().get("_id"))
    return {
        "id": str(new_user._id), 
        "username": new_user.username,
        "email": new_user.email,
        "role": new_user.role
    }


def update_label_user(label, id):
    user = UserEntity.find_by_id(id)
    if user:
        user.label = label
        user.save()
        return user.to_dictionary()
    return None

def get_user_by_id(id):
    user = UserEntity.find_by_id(id)
    if user:
        return user.to_dictionary()
    return None

