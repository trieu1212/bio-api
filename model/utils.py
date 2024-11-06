import cv2
import numpy as np

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