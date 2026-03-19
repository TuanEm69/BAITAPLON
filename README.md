  # BAITAPLON

  # Bước 1.Lấy dataset và train AI
  - Truy cập trang web kaggle.com để lấy API KEY, tải file kaglle.json về
  - Tải dataset trực tiếp từ Kaggle thông qua API (ifeanyinneji/nike-adidas... và hasibalmuzdadid/shoe-vs-sandal...).
  - Truy cập trang web google colab.com, tạo 1 sổ tay mới, thay đổi thời gian chạy thành GPU T4 và dán đoạn code này lên 1 cell:
- **Code:** 
import os
import shutil
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from google.colab import files

if not os.path.exists('/root/.kaggle/kaggle.json'):
    print("--- HÃY CHỌN FILE KAGGLE.JSON ---")
    uploaded = files.upload()
    !mkdir -p ~/.kaggle
    !cp kaggle.json ~/.kaggle/
    !chmod 600 ~/.kaggle/kaggle.json

print("\n--- ĐANG TẢI DỮ LIỆU ---")

!kaggle datasets download -d ifeanyinneji/nike-adidas-shoes-for-image-classification-dataset
!unzip -o -q nike-adidas-shoes-for-image-classification-dataset.zip -d data_brand

!kaggle datasets download -d hasibalmuzdadid/shoe-vs-sandal-vs-boot-dataset-15k-images
!unzip -o -q shoe-vs-sandal-vs-boot-dataset-15k-images.zip -d data_type

FINAL_DATA = '/content/final_dataset'
os.makedirs(FINAL_DATA, exist_ok=True)


def merge_data(src, label_name):
    dest = os.path.join(FINAL_DATA, label_name)
    if not os.path.exists(dest):
        shutil.copytree(src, dest)
    print(f"Đã gộp nhãn: {label_name}")

merge_data('/content/data_brand/train/adidas', 'Adidas')
merge_data('/content/data_brand/train/nike', 'Nike')
merge_data('/content/data_type/Shoe vs Sandal vs Boot Dataset/Boot', 'Boot')
merge_data('/content/data_type/Shoe vs Sandal vs Boot Dataset/Sandal', 'Sandal')
merge_data('/content/data_type/Shoe vs Sandal vs Boot Dataset/Shoe', 'Shoe_Normal')

datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    FINAL_DATA, target_size=(224, 224), batch_size=32,
    class_mode='categorical', subset='training'
)

val_gen = datagen.flow_from_directory(
    FINAL_DATA, target_size=(224, 224), batch_size=32,
    class_mode='categorical', subset='validation'
)

print("\n--- ĐANG HUẤN LUYỆN MODEL TỔNG HỢP ---")
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(train_gen.num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, validation_data=val_gen, epochs=10)
model_file = 'model_shoe_expert_v2.h5'
model.save(model_file)

with open('labels.txt', 'w') as f:
    indices = sorted(train_gen.class_indices.items(), key=lambda x: x[1])
    for name, idx in indices:
        f.write(f"{idx}: {name}\n")

print(f"\n--- XONG! ĐANG TẢI {model_file} ---")
files.download(model_file)
files.download('labels.txt')

   

- file model_shoe_and_brand.h5 sẽ tự động được tải về.

  # Bước 2. Thiết kế giao diện và liên kiết
  - Mở virual studio code, tạo 1 folder đặt tên dự án là BTL_TTTNT
  - thêm 1 file tên app.py
  - di chuyển file model_shoe_and_brand.h5 đã tải vào folder BTL_TTTNT
  - thiết lập code tạo giao diện và liên kết dataset. Tại file code nằm trong thư mục "app.py" up trên githup, ccopy code và dán vào file app.py trong máy và chạy
  - Chạy file app.py với lệnh streamlit run app.py
  
  
