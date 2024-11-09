import numpy as np
from keras._tf_keras.keras.preprocessing import image
from keras._tf_keras.keras.models import load_model
from deepface import DeepFace
# Load model và label_map
img1_path = "/home/trieu/project/bio-python/data/faces/672add53430a0755e5ee8835_trieu/17.jpg"
img2_path = "/home/trieu/project/bio-python/data/faces/672add53430a0755e5ee8835_trieu/12.jpg"

img3_path = "/mnt/c/Users/Hello/Downloads/tesst4.jpg"
img4_path = "/mnt/c/Users/Hello/Downloads/test3.jpg"

result = DeepFace.verify(
  img1_path = img4_path,
  img2_path = img1_path,
  model_name = "Facenet",
  enforce_detection=False
)


print(result)