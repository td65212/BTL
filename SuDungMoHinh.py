import os
import numpy as np
from keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model

# Danh sách cảm xúc
class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# 📌 Chọn mô hình cần sử dụng (ANN hoặc CNN)
model_path = "/content/emotion_ann_model.keras"  # Nếu dùng ANN

# Tải mô hình đã lưu
if not os.path.exists(model_path):
    print(f"LỖI: Không tìm thấy mô hình tại {model_path}. Kiểm tra lại!")
    exit()

model = load_model(model_path)
print("✅ Mô hình đã tải thành công!")

# Hàm dự đoán cảm xúc từ ảnh đầu vào
def predict_emotion(img_path):
    if not os.path.exists(img_path):
        print(f"❌ LỖI: Không tìm thấy ảnh tại {img_path}. Kiểm tra lại!")
        return

    # Tiền xử lý ảnh
    img_size = (48, 48)
    img = load_img(img_path, target_size=img_size, color_mode="grayscale")
    img_array = img_to_array(img) / 255.0  # Chuẩn hóa dữ liệu
    img_array = np.expand_dims(img_array, axis=0)  # Thêm batch dimension

    # Dự đoán với mô hình đã lưu
    prediction = model.predict(img_array)

    # Lấy nhãn cảm xúc có xác suất cao nhất
    class_idx = np.argmax(prediction)
    predicted_emotion = class_labels[class_idx]

    print(f"🎭 Cảm xúc dự đoán: {predicted_emotion}")

# 📌 Nhập đường dẫn ảnh từ người dùng và dự đoán
img_path = input("🖼 Nhập đường dẫn ảnh cần dự đoán: ")
predict_emotion(img_path)
