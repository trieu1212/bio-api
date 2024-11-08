import numpy as np
from keras._tf_keras.keras.preprocessing import image
from keras._tf_keras.keras.models import load_model
from deepface import DeepFace
# Load model và label_map
img1_path = "/home/trieu/project/bio-python/data/faces/672add53430a0755e5ee8835_trieu/17.jpg"
img2_path = "/home/trieu/project/bio-python/data/faces/672add53430a0755e5ee8835_trieu/18.jpg"
result = DeepFace.verify(
  img1_path = img1_path,
  img2_path = img2_path,
  model_name = "Facenet",
)

print(result)