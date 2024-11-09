import cv2
import numpy as np
import pickle
import os
from deepface import DeepFace
from config import Config

PREPROCCESS_DIR = Config.PREPROCESS_DIR
FACES_DIR = Config.FACES_DIR
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


def preprocess_image(id, username):
    input_dir = os.path.join('data/faces', f"{id}_{username}")
    image_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

    user_dir = os.path.join(PREPROCCESS_DIR, f"{id}_{username}")
    os.makedirs(user_dir, exist_ok=True)

    existing_files = os.listdir(user_dir)
    existing_files = [f for f in existing_files if f.endswith('.jpg')]
    existing_indices = [int(f.split('.')[0]) for f in existing_files]  # Chỉnh sửa cách lấy chỉ số
    next_index = max(existing_indices, default=0) + 1  # Lấy chỉ số tiếp theo

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
                            
                        # Lưu ảnh khuôn mặt vào tệp với tên tiếp theo
                        image_path = os.path.join(user_dir, f"{next_index}.jpg")
                        if cv2.imwrite(image_path, face_img):
                            print(f"Đã lưu ảnh khuôn mặt vào {image_path}")
                            next_index += 1  # Cập nhật lại chỉ số cho ảnh tiếp theo
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
    existing_indices = [int(f.split('.')[0]) for f in existing_files]  # Chỉnh sửa cách lấy chỉ số
    next_index = max(existing_indices, default=0) + 1  # Lấy chỉ số tiếp theo

    for i, image_file in enumerate(images):
        try:
            img = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                print(f"Lỗi khi đọc ảnh {i + 1}")
                continue

            # Lưu ảnh vào tệp với tên tiếp theo
            image_path = os.path.join(user_dir, f"{next_index}.jpg")
            if cv2.imwrite(image_path, img):
                print(f"Đã lưu ảnh vào {image_path}")
                next_index += 1  # Cập nhật lại chỉ số cho ảnh tiếp theo
            else:
                print(f"Lỗi khi lưu ảnh vào {image_path}")
        except Exception as e:
            print(f"Lỗi khi xử lý ảnh {i + 1}: {e}")
            continue