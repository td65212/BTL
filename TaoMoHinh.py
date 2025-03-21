import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# --- BƯỚC 0: GIẢI NÉN DỮ LIỆU ---
zip_file = "/content/FER-2013.zip"
extract_dir = "dataset"

if not os.path.exists(extract_dir):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print("Dữ liệu đã được giải nén!")


# --- BƯỚC 1: TIỀN XỬ LÝ DỮ LIỆU ---
img_size = (48, 48)
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    "dataset/train",
    target_size=img_size,
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="categorical",
    subset="training"
)
val_generator = train_datagen.flow_from_directory(
    "dataset/train",
    target_size=img_size,
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="categorical",
    subset="validation"
)

# --- BƯỚC 2: XÂY DỰNG MÔ HÌNH ANN ---
model = Sequential([
    Flatten(input_shape=(48, 48, 1)),  # Chuyển ảnh thành vector 2304 phần tử
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(train_generator.num_classes, activation='softmax')  # Output layer
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- BƯỚC 3: HUẤN LUYỆN MÔ HÌNH ---
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    verbose=1
)

# --- BƯỚC 4: ĐÁNH GIÁ MÔ HÌNH ---
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    "dataset/test",
    target_size=img_size,
    batch_size=batch_size,
    color_mode="grayscale",
    class_mode="categorical"
)

loss, accuracy = model.evaluate(test_generator)
print(f"Độ chính xác trên tập kiểm tra: {accuracy:.2%}")

# --- BƯỚC 5: LƯU MÔ HÌNH ---
model.save("emotion_ann_model.keras")
print("Mô hình ANN đã được lưu thành công!")
