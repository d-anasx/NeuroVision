
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

from PIL import Image

# Page configuration
st.set_page_config(
    page_title="NeuroVision AI",
    page_icon="🧠",
    layout="centered"
)

# Model and classes
MODEL_PATH = "best_model.keras"

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

# Load the trained model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()

# App interface
st.title("🧠 NeuroVision AI")
st.subheader("Brain MRI Classification")
st.write(
    "Upload a brain MRI image to obtain a prediction "
    "from the trained four-class CNN."
)

uploaded_file = st.file_uploader(
    "Choose a brain MRI image",
    type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file is not None:
    try:
        # Read and display the uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded MRI", width="stretch")

        # Convert RGB image to OpenCV BGR format
        image_array = np.array(image)
        image_bgr = cv2.cvtColor(
            image_array, cv2.COLOR_RGB2BGR
        )

        # Preprocess exactly as during training
        image_bgr = cv2.resize(image_bgr, (224, 224))
        image_bgr = image_bgr.astype(np.float32) / 255.0

        # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
        input_image = np.expand_dims(image_bgr, axis=0)

        # Predict
        probabilities = model.predict(
            input_image, verbose=0
        )[0]

        predicted_index = int(np.argmax(probabilities))
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(probabilities[predicted_index])

        st.divider()
        st.subheader("Prediction")

        st.success(f"Predicted class: {predicted_class}")
        st.metric("Model confidence", f"{confidence:.2%}")

        st.subheader("Class probabilities")

        for i, class_name in enumerate(CLASS_NAMES):
            probability = float(probabilities[i])
            st.write(f"**{class_name}** — {probability:.2%}")
            st.progress(probability)

        st.warning(
            "This model is an educational prototype, not a "
            "medical diagnostic tool. Its predictions must not "
            "be used to make medical decisions."
        )

    except Exception as e:
        st.error(f"Could not process this image: {e}")
