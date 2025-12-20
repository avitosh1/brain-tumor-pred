import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ================= CONFIG =================
IMG_SIZE = (224, 224)
MODEL_PATH = "models/efficientnet.h5"

st.set_page_config(
    page_title="Brain Tumor MRI Classification",
    layout="centered"
)

# ================= LOAD MODEL =================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# ================= CLASS NAMES =================
# ⚠️ MUST match training order exactly
class_names = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

# ================= UI =================
st.title("🧠 Brain Tumor MRI Classification")
st.write(
    "Upload a brain MRI image and the AI model will predict the tumor type."
)

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

# ================= IMAGE PROCESSING =================
def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# ================= PREDICTION =================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    processed_image = preprocess_image(image)

    with st.spinner("Analyzing MRI image..."):
        predictions = model.predict(processed_image)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

    st.success("Prediction Complete")

    st.subheader("🧬 Prediction Result")
    st.write(f"**Tumor Type:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    # Optional probability breakdown
    st.subheader("📊 Class Probabilities")
    for i, cls in enumerate(class_names):
        st.write(f"{cls}: {predictions[0][i]*100:.2f}%")

# ================= FOOTER =================
st.markdown("---")
st.caption(
    "AI-assisted Brain Tumor Classification | Deep Learning + EfficientNet"
)
