from bson.objectid import ObjectId
from pymongo import MongoClient
from config import Config


users_collection = MongoClient(Config.MONGO_URI)[Config.MONGO_DB_NAME]["users"]

print(users_collection)
class UserEntity:
    def __init__(self, username, password, email, role, label = None, _id=None):
        self._id = _id or ObjectId()
        self.username = username
        self.password = password
        self.email = email
        self.role = role
        self.label = label

    def to_dictionary(self):
         return {
            "username": self.username,
            "password": self.password,
            "email": self.email,
            "role": self.role,
            "label": self.label,
            "_id": self._id
        }
    
    def save(self):
        if self._id is None:
            user_data = self.to_dictionary()
            result = users_collection.insert_one(user_data)
            self._id = result.inserted_id
        else:
            users_collection.replace_one({"_id": self._id}, self.to_dictionary(), upsert=True)

    
    @staticmethod
    def find_by_label(label):
        user = users_collection.find_one({"label": label})
        if user:
            return UserEntity(user["_id"], user["username"], user["password"], user["email"], user["role"], user["label"])
        return None
    
    @staticmethod
    def find_by_id(id):
        user = users_collection.find_one({"_id": ObjectId(id)})
        if user:
            return UserEntity(user["_id"], user["username"], user["password"], user["email"], user["role"], user["label"])
        return None
    
    @staticmethod
    def find_by_username(username):
        user = users_collection.find_one({"username": username})
        if user:
            return UserEntity(user["_id"], user["username"], user["password"], user["email"], user["role"], user["label"])
        return None
    
    @staticmethod
    def update_password(id, password):
        users_collection.update_one({"_id": ObjectId(id)}, {"$set": {"password": password}})

    @staticmethod
    def delete(id):
        users_collection.delete_one({"_id": ObjectId(id)})

    