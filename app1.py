import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import os

from tensorflow.keras.applications.efficientnet import preprocess_input

# ==================================================
# CONFIG
# ==================================================

IMG_SIZE = (224, 224)
MODEL_PATH = "models/efficientnet_final.keras"

st.set_page_config(
    page_title="Brain Tumor MRI Classification",
    page_icon="🧠",
    layout="wide"
)

# ==================================================
# LOAD MODEL
# ==================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# ==================================================
# LOAD CLASS NAMES
# ==================================================

try:
    class_names = np.load(
        "models/class_names.npy",
        allow_pickle=True
    )
except Exception as e:
    st.error(f"Error loading class names: {e}")
    st.stop()

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("🧠 Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "🏠 Home",
        "🔍 MRI Prediction",
        "📚 Model Overview",
        "📊 Results Dashboard"
    ]
)

st.sidebar.markdown("---")

st.sidebar.subheader("Project Details")
st.sidebar.write("Model: EfficientNetB0")
st.sidebar.write("Framework: TensorFlow / Keras")
st.sidebar.write("Classes: 4")
st.sidebar.write("Image Size: 224 × 224")

# ==================================================
# HOME PAGE
# ==================================================

if page == "🏠 Home":

    st.title("🧠 Brain Tumor MRI Classification System")

    st.markdown("""
    ### Deep Learning Based Brain Tumor Detection

    This project classifies Brain MRI scans into:

    - Glioma
    - Meningioma
    - No Tumor
    - Pituitary

    The final deployed model uses EfficientNetB0 Transfer Learning.
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Test Accuracy", "86.59%")

    with col2:
        st.metric("Classes", "4")

    with col3:
        st.metric("Model", "EfficientNetB0")

    sample_path = "results/sample_images.png"

    if os.path.exists(sample_path):
        st.image(
            sample_path,
            caption="Sample MRI Images",
            width=800
        )

# ==================================================
# MRI PREDICTION PAGE
# ==================================================

elif page == "🔍 MRI Prediction":

    st.title("🔍 Brain Tumor Prediction")

    uploaded_file = st.file_uploader(
        "Upload MRI Image",
        type=["jpg", "jpeg", "png"]
    )

    def preprocess_image(image):
        image = image.resize(IMG_SIZE)
        image = np.array(image).astype(np.float32)
        image = preprocess_input(image)
        image = np.expand_dims(image, axis=0)
        return image

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.image(
                image,
                caption="Uploaded MRI",
                width=500
            )

        with col2:
            st.subheader("Image Details")
            st.write(f"Width: {image.size[0]} px")
            st.write(f"Height: {image.size[1]} px")
            st.write(f"Mode: {image.mode}")

        if st.button("Predict Tumor"):

            processed_image = preprocess_image(image)

            predictions = model.predict(
                processed_image,
                verbose=0
            )

            predicted_index = np.argmax(predictions)

            predicted_class = class_names[predicted_index]

            confidence = float(
                np.max(predictions) * 100
            )

            st.success("Prediction Completed")

            c1, c2 = st.columns(2)

            with c1:
                st.metric(
                    "Predicted Class",
                    predicted_class
                )

            with c2:
                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

            st.subheader("Confidence Level")

            st.progress(
                min(int(confidence), 100)
            )

            probability_df = pd.DataFrame({
                "Tumor Type": class_names,
                "Probability (%)": predictions[0] * 100
            })

            st.subheader(
                "Class Probability Distribution"
            )

            st.bar_chart(
                probability_df.set_index(
                    "Tumor Type"
                )
            )

            st.subheader(
                "Detailed Probabilities"
            )

            for i, cls in enumerate(class_names):
                st.write(
                    f"**{cls}** : {predictions[0][i] * 100:.2f}%"
                )

            st.warning(
                "Educational project only. Not a medical diagnosis."
            )

# ==================================================
# MODEL OVERVIEW PAGE
# ==================================================

elif page == "📚 Model Overview":

    st.title("📚 Model Overview")

    with st.expander("Custom CNN"):

        st.markdown("""
        ### Custom CNN

        Architecture:

        - Conv2D(64)
        - Batch Normalization
        - Max Pooling
        - Conv2D(128)
        - Batch Normalization
        - Max Pooling
        - Conv2D(256)
        - Batch Normalization
        - Max Pooling
        - Dense(256)
        - Dropout(0.5)

        Used as the baseline model.
        """)

    with st.expander("EfficientNetB0"):

        st.markdown("""
        ### EfficientNetB0

        Transfer Learning Model

        Pretrained on ImageNet

        Advantages:

        - Better feature extraction
        - Faster convergence
        - Higher accuracy
        - Strong generalization

        Final Test Accuracy: 86.59%
        """)

# ==================================================
# RESULTS DASHBOARD
# ==================================================

elif page == "📊 Results Dashboard":

    st.title("📊 Results Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Custom CNN Accuracy",
            "21.95%"
        )

    with col2:
        st.metric(
            "EfficientNet Accuracy",
            "86.59%"
        )

    st.markdown("---")

    st.header("📈 Model Comparison")

    comparison_path = "results/model_comparison.png"

    if os.path.exists(comparison_path):
        st.image(
            comparison_path,
            width=1000
        )

    st.header("🧩 EfficientNet Confusion Matrix")

    cm_path = "results/efficientnet_confusion_matrix.png"

    if os.path.exists(cm_path):
        st.image(
            cm_path,
            width=800
        )

    st.header("🔥 Class-wise Performance")

    heatmap_path = "results/classwise_metrics_heatmap.png"

    if os.path.exists(heatmap_path):
        st.image(
            heatmap_path,
            width=800
        )

    st.header("📋 Metrics Table")

    metrics_path = "results/metrics.csv"

    if os.path.exists(metrics_path):
        metrics_df = pd.read_csv(metrics_path)

        st.dataframe(
            metrics_df,
            width=800
        )

# ==================================================
# FOOTER
# ==================================================

st.markdown("---")

st.caption(
    "Brain Tumor MRI Classification using Deep Learning and EfficientNetB0"
)
