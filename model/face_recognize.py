from curses.ascii import EM
import os
import cv2
import pickle
import random
import numpy as np
from deepface import DeepFace

from keras._tf_keras.keras.models import load_model
# from config import Config
from sklearn.metrics.pairwise import cosine_similarity
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '../data/faces')
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

def train_embeddings(user_id, username, images,embeddings_path=EMBEDDINGS_PATH):
    embeddings = []
    user_folder = f"{user_id}_{username}"
    user_dir = os.path.join(DATA_DIR, user_folder)
    os.makedirs(user_dir, exist_ok=True)

    for i, image_file in enumerate(images): 
        try:
            img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                print(f"Lỗi khi đọc ảnh {i + 1}")
                continue

            # augmented_img = augment_image(img)
            embedding = detect_and_extract_embedding(img)
            
            if embedding is not None:
                embeddings.append(normalize_embedding(embedding))
                image_path = os.path.join(user_dir, f"{i + 1}.jpg")
                cv2.imwrite(image_path, img)
        except Exception as e:
            print(f"Lỗi khi xử lý ảnh {i + 1}: {e}")
            continue

    if embeddings:
        average_embedding = calculate_average_embedding(embeddings)
        embeddings = [average_embedding]

    embeddings_file = os.path.join(embeddings_path, f"{user_folder}.pkl") 
    with open(embeddings_file, 'wb') as f:
        pickle.dump(embeddings, f)

    return user_folder, embeddings

def normalize_embedding(embedding):
    return embedding / np.linalg.norm(embedding)

def calculate_average_embedding(embeddings):
    average_embedding = np.mean(embeddings, axis=0)
    return average_embedding / np.linalg.norm(average_embedding)

def augment_image(img):
    # Rotate
    angle = random.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((img.shape[1] // 2, img.shape[0] // 2), angle, 1)
    img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    # Brightness adjustment
    brightness = random.uniform(0.8, 1.2)
    img = np.clip(img * brightness, 0, 255).astype(np.uint8)

    # Add Gaussian blur
    if random.choice([True, False]):
        img = cv2.GaussianBlur(img, (5, 5), 0)

    # Flip horizontally
    if random.choice([True, False]):
        img = cv2.flip(img, 1)
    
    # Shear Transformation (optional)
    if random.choice([True, False]):
        shear = random.uniform(-0.2, 0.2)
        M = np.float32([[1, shear, 0], [0, 1, 0]])
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    return img