import numpy as np
from keras._tf_keras.keras.preprocessing import image
from keras._tf_keras.keras.models import load_model
import json
# Load model và label_map
model = load_model('saved_models/model.keras')
with open('label_map.json', 'r') as f:
    label_map = json.load(f)

print(label_map)
label_map_list = [label_map[key] for key in sorted(label_map.keys())]
# Load ảnh mẫu
img_path = '/home/trieu/project/bio-python/data/faces/672a2d43ccca1ceafbcdc352_trump/1.jpg'  # Thay đổi đường dẫn tới ảnh của bạn
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = x / 255.0

# Dự đoán
predictions = model.predict(x)

# In ra kết quả
predicted_class = np.argmax(predictions)
predicted_label = label_map_list[predicted_class]

print(f"Dự đoán: {predicted_class} ({predicted_label})")
print(f"Xác suất cho từng lớp: {predictions}")