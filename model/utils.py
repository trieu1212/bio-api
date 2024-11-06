import cv2
import numpy as np
import pickle

def predict_face(model, image):
    image = preprocess_image(image)
    image = np.expand_dims(image, axis=0)  
    predictions = model.predict(image)
    print(predictions)  
    predicted_label = np.argmax(predictions)  
    return predicted_label


def predict_face_with_probabilities(model, image):
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)
    print("Predictions:", predictions)  
    predicted_label = np.argmax(predictions)
    return predicted_label

def preprocess_image(image):
    image = cv2.resize(image, (224, 224))  
    image = image / 255.0  
    return image


def recognize_face(image_path, embeddings_path, threshold): 
    with open(embeddings_path, 'rb') as f:
        known_embeddings, labels, label_map = pickle.load(f)

    img = cv2.imread(image_path)
    if img is None:
        return None, None

    user, similarity = verify_face(img, known_embeddings, threshold)


    if user == "unknown" or similarity < threshold:
        return "unknown", similarity

    return user, similarity