import cv2
import numpy as np
import os
import math
from deepface import DeepFace
from config import Config

PREPROCCESS_DIR = Config.PREPROCESS_DIR
FACES_DIR = Config.FACES_DIR

def preprocess_image(id, username):
    input_dir = os.path.join('data/faces', f"{id}_{username}")
    image_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    user_dir = os.path.join(PREPROCCESS_DIR, f"{id}_{username}")
    os.makedirs(user_dir, exist_ok=True)

    existing_files = os.listdir(user_dir)
    existing_files = [f for f in existing_files if f.endswith('.jpg')]
    existing_indices = [int(f.split('.')[0]) for f in existing_files]  
    next_index = max(existing_indices, default=0) + 1 

    for i, image_file in enumerate(image_files):
        try:
            image_data = cv2.imread(image_file)
            if image_data is None:
                print(f"Lỗi khi đọc ảnh {image_file}")
                continue

            detected_faces = DeepFace.extract_faces(image_data, detector_backend='retinaface', enforce_detection=False)
            if detected_faces:
                for j, face in enumerate(detected_faces):
                    face_img = face["face"]

                    if face_img.shape[:2] != (160, 160):
                        face_img = cv2.resize(face_img, (160, 160))
                        face_img = (face_img * 255).astype(np.uint8)
                        if len(face_img.shape) == 2:  
                            face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
                            
                        image_path = os.path.join(user_dir, f"{next_index}.jpg")
                        if cv2.imwrite(image_path, face_img):
                            print(f"Đã lưu ảnh khuôn mặt vào {image_path}")
                            next_index += 1  
                        else:
                            print(f"Lỗi khi lưu ảnh khuôn mặt vào {image_path}")
                    else:
                        print(f"Lỗi khi xử lý khuôn mặt từ ảnh {i + 1}")
            else:
                print(f"Không phát hiện khuôn mặt trong ảnh {i + 1}")
        
        except Exception as e:
            print(f"Lỗi khi xử lý ảnh {i + 1}: {e}")

def save_user_pics(id, username, images):
    user_dir = os.path.join(FACES_DIR, f"{id}_{username}")
    os.makedirs(user_dir, exist_ok=True)

    existing_files = os.listdir(user_dir)
    existing_files = [f for f in existing_files if f.endswith('.jpg')]
    existing_indices = [int(f.split('.')[0]) for f in existing_files]  
    next_index = max(existing_indices, default=0) + 1  

    for i, image_file in enumerate(images):
        try:
            img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                print(f"Lỗi khi đọc ảnh {i + 1}")
                continue

            image_path = os.path.join(user_dir, f"{next_index}.jpg")
            if cv2.imwrite(image_path, img):
                print(f"Đã lưu ảnh vào {image_path}")
                next_index += 1  
            else:
                print(f"Lỗi khi lưu ảnh vào {image_path}")
        except Exception as e:
            print(f"Lỗi khi xử lý ảnh {i + 1}: {e}")
            continue


def cosine_distance(x1, x2):
    mag1 = 0.0
    mag2 = 0.0
    product = 0.0

    for i in range(len(x1)):
        mag1 += x1[i] ** 2
        mag2 += x2[i] ** 2
        product += x1[i] * x2[i]

    mag1 = math.sqrt(mag1)
    mag2 = math.sqrt(mag2)

    result = product / (mag1 * mag2)
    print(f"Result: {result}")

    return result