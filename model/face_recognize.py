from curses.ascii import EM
import os
import cv2
import pickle
from mtcnn import MTCNN
import numpy as np
from deepface import DeepFace

from keras._tf_keras.keras.models import load_model
# from config import Config
from sklearn.metrics.pairwise import cosine_similarity
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '../data/faces')
MODEL_PATH = 'saved_models/facenet_keras.h5'
EMBEDDINGS_PATH = 'embeddings.pickle'
THRESHOLD = 0.7

# face_model = load_model(MODEL_PATH) 

detector = MTCNN()

def get_embedding(img):
    return DeepFace.represent(img, model_name = 'Facenet', enforce_detection=False)

def detect_and_extract_embedding(img):
        try:
            results = detector.detect_faces(img)
            if results:
                x1, y1, width, height = results[0]['box']
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + width, y1 + height
                face = img[y1:y2, x1:x2]
                return get_embedding(face)
            else:
                return None

        except Exception as e:
            return None

def train_embeddings(data_dir=DATA_DIR, embeddings_path=EMBEDDINGS_PATH):
    embeddings = {}
    for user_folder in os.listdir(data_dir):
        user_folder_path = os.path.join(data_dir, user_folder)
        if os.path.isdir(user_folder_path):
            embeddings[user_folder] = []
            for filename in os.listdir(user_folder_path):
                if filename.endswith(('.jpg', '.png', '.jpeg')):
                    image_path = os.path.join(user_folder_path, filename)
                    try:
                        img = cv2.imread(image_path)
                        if img is None: continue
                        emb = detect_and_extract_embedding(img)
                        if emb is not None: 
                           embeddings[user_folder].append(emb)
                    except Exception as e:
                        print(f"Lỗi khi xử lý ảnh {image_path}: {e}")

    with open(embeddings_path, 'wb') as f:
        pickle.dump(embeddings, f)

def verify_face(image, known_embeddings, threshold=THRESHOLD):
    try:
        embedding = get_embedding(image) 
    except IndexError:
        return "unknown", 0

    best_match = None
    highest_similarity = 0

    for user, known_embedding_list in known_embeddings.items():
        for known_embedding in known_embedding_list:
            similarity = cosine_similarity([embedding], [known_embedding])[0][0] 
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = user

    if highest_similarity > threshold:
        return best_match, highest_similarity
    else:
        return "unknown", highest_similarity

