from curses.ascii import EM
import os
import cv2
import pickle
from mtcnn import MTCNN
import numpy as np

from keras._tf_keras.keras.models import load_model
# from config import Config
from sklearn.metrics.pairwise import cosine_similarity
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '../data/faces')
MODEL_PATH = 'facenet_keras.h5'
EMBEDDINGS_PATH = 'embeddings.pickle'
THRESHOLD = 0.7

face_model = load_model(MODEL_PATH) 

detector = MTCNN()

def get_embedding(face_pixels):  
    face_pixels = np.asarray(face_pixels)
    face_pixels = cv2.resize(face_pixels, (160, 160))
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = np.expand_dims(face_pixels, axis=0)
    yhat = face_model.predict(samples)
    return yhat[0]

def detect_and_extract_embeddings():
    embeddings = []
    labels = []
    label_map = {}
    label_counter = 0

    for person_dir in os.listdir(DATA_DIR):
        person_path = os.path.join(DATA_DIR, person_dir)
        if os.path.isdir(person_path):
            if person_dir not in label_map:
                label_map[person_dir] = str(label_counter) 
                label_counter += 1
            for img_file in os.listdir(person_path):
                try: 
                    img = cv2.imread(os.path.join(person_path, img_file))
                    if img is None: continue 
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    results = detector.detect_faces(img)
                    if results:
                        x, y, w, h = results[0]['box']
                        face = img[y:y+h, x:x+w]
                        emb = get_embedding(face)
                        embeddings.append(emb)
                        labels.append(label_map[person_dir])
                except Exception as e:
                    print(f"Lỗi khi xử lý ảnh {os.path.join(person_path, img_file)}: {e}")

    with open(EMBEDDINGS_PATH, 'wb') as f:
        pickle.dump((embeddings, labels, label_map), f)

    return embeddings, labels, label_map

def train_embeddings():
    detect_and_extract_embeddings()
    print('train successfully!')

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

