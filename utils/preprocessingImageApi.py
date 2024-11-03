import numpy as np
import cv2


def preprocess_image(image):
    resized_image = cv2.resize(image, (128, 128))
    img_array = np.array(resized_image, dtype='float32')  
    img_array = img_array / 255.0  
    img_array = np.expand_dims(img_array, axis=0) 

    return img_array