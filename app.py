import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import os

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN
# ==========================================
st.set_page_config(
    page_title="Hệ thống AI Nhận diện Giày & Thương hiệu",
    page_icon="👟",
    layout="centered"
)

# Tên file model mới (đảm bảo file này nằm trong thư mục btl_ttnt)
MODEL_NAME = 'model_shoe_and_brand.h5' 

# Danh sách nhãn đầy đủ (Phải khớp thứ tự với Model gộp)
class_names = ['Adidas (Thương hiệu)', 'Boot (Giày cổ cao)', 'Nike (Thương hiệu)', 'Sandal (Dép quai hậu)', 'Shoe (Giày thể thao)']

# ==========================================
# 2. HÀM TẢI MODEL
# ==========================================
@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_NAME):
        return None, f"❌ Không tìm thấy file '{MODEL_NAME}'!"
    try:
        model = tf.keras.models.load_model(MODEL_NAME)
        return model, None
    except Exception as e:
        return None, f"⚠️ Lỗi load model: {str(e)}"

model, error = load_my_model()

# ==========================================
# 3. GIAO DIỆN NGƯỜI DÙNG (UI)
# ==========================================
st.title("👟 AI nhận diện Giày & Thương hiệu")
st.write("Ứng dụng có thể phân biệt: **Nike, Adidas, Boot, Sandal và Shoe**.")

if model is None:
    st.error(error)
    st.stop()

uploaded_file = st.file_uploader("Tải ảnh giày lên...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Hiển thị ảnh
    image = Image.open(uploaded_file)
    st.image(image, caption='Ảnh đã tải lên', use_container_width=True)
    
    st.write("---")
    
    with st.spinner('Đang phân tích thương hiệu và kiểu dáng...'):
        # Tiền xử lý ảnh
        size = (224, 224)
        image_resized = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
        img_array = np.asarray(image_resized)
        
        # Rescale về [0, 1] và thêm chiều batch
        img_reshape = img_array[np.newaxis, ...] / 255.0

        # Dự đoán
        prediction = model.predict(img_reshape)
        label_idx = np.argmax(prediction)
        confidence = np.max(prediction) * 100

        # Hiển thị kết quả rực rỡ
        st.subheader(f"Kết quả: {class_names[label_idx]}")
        
        # Hiển thị xác suất chi tiết
        if confidence > 80:
            st.success(f"Độ tin cậy: {confidence:.2f}%")
        else:
            st.warning(f"Độ tin cậy: {confidence:.2f}% (Kết quả có thể chưa chính xác)")

        # Biểu đồ phân tích các nhãn còn lại
        st.write("Phân tích xác suất chi tiết:")
        chart_data = {class_names[i]: float(prediction[0][i]) for i in range(len(class_names))}
        st.bar_chart(chart_data)

st.markdown("---")
st.caption("Phiên bản cập nhật nhận diện Thương hiệu v2.0")