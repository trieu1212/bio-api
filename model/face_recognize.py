from curses.ascii import EM
import os
import cv2
import pickle
import random
import numpy as np

from deepface import DeepFace

from model.utils import save_user_pics, preprocess_image
from keras._tf_keras.keras.models import load_model
from config import Config
from sklearn.metrics.pairwise import cosine_similarity
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREPROCESS_DIR = Config.PREPROCESS_DIR
FACES_DIR = Config.FACES_DIR
MODEL_PATH = 'saved_models/facenet_keras.h5'
EMBEDDINGS_PATH = 'embeddings'
THRESHOLD = 0.7


def get_embedding(img):
    return DeepFace.represent(img, model_name = 'Facenet', enforce_detection=False)

def detect_and_extract_embedding(img):
    try:
        detected_faces = DeepFace.extract_faces(img, detector_backend='retinaface')
        if detected_faces:
            face_img = detected_faces[0]["face"]
            if face_img.shape[:2] != (160, 160):
                face_img = cv2.resize(face_img, (160, 160))
            print(detected_faces)
            embedding_obj = get_embedding(face_img)
            return np.array(embedding_obj[0]["embedding"], dtype=float) if embedding_obj else None
        else:
            print("Không phát hiện khuôn mặt.")
            return None
    except Exception as e:
        print(f"Lỗi khi phát hiện hoặc trích xuất embedding: {e}")
        return None

def train_embeddings(user_id, username, images, embeddings_path=EMBEDDINGS_PATH):
    embeddings = []

    save_user_pics(user_id, username, images)
    preprocess_image(user_id, username)

    preprocessed_user_dir = os.path.join(PREPROCESS_DIR, f"{user_id}_{username}")

    for face_file in os.listdir(preprocessed_user_dir):
        face_path = os.path.join(preprocessed_user_dir, face_file)
        if face_file.lower().endswith(('png', 'jpg', 'jpeg')):
            face_img = cv2.imread(face_path)

        if face_img is not None:
            embedding_obj = DeepFace.represent(face_img, model_name='Facenet', enforce_detection=False)
            if embedding_obj:
                embedding = np.array(embedding_obj[0]["embedding"], dtype=float)
                embeddings.append(normalize_embedding(embedding))
        else:
            print(f"Không thể đọc file ảnh: {face_file}")

    if embeddings:
        average_embedding = calculate_average_embedding(embeddings)
        embeddings = [average_embedding]
        
        embeddings_file = os.path.join(embeddings_path, f"{user_id}_{username}.pkl")
        os.makedirs(embeddings_path, exist_ok=True)
        with open(embeddings_file, 'wb') as f:
            pickle.dump(embeddings, f)
        print(f"Đã lưu embeddings vào {embeddings_file}")
    else:
        print("Không có embeddings để lưu.")

    return f"{user_id}_{username}", embeddings

def normalize_embedding(embedding):
    return embedding / np.linalg.norm(embedding)

def calculate_average_embedding(embeddings):
    average_embedding = np.mean(embeddings, axis=0)
    return average_embedding / np.linalg.norm(average_embedding)
