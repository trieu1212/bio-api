from api.entity.userEntity import UserEntity


def create_user(user_data):
    new_user = UserEntity(
        username=user_data["username"],
        password=user_data["password"],
        email=user_data["email"],
        role=user_data["role"]
    )
    new_user.save()
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
        user.save(update=True)
        user_dict = user.to_dictionary()
        id, username, email, role, label = user.to_dictionary().get("_id"), user.to_dictionary().get("username"), user.to_dictionary().get("email"), user.to_dictionary().get("role"), user.to_dictionary().get("label")
        return user_dict
    return None

def get_user_by_id(id):
    user = UserEntity.find_by_id(id)
    if user:
        return user.to_dictionary()
    return None

def get_user_by_label(label):
    user = UserEntity.find_by_label(label)
    if user:
        return user.to_dictionary()
    return None