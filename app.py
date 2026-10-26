
import os

import mlflow
import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image


# -----------------------------
# Configuration
# -----------------------------

MLFLOW_TRACKING_URI = (
    "https://billion-rides-keno-viewers.trycloudflare.com"
)

MODEL_URI = "models:/NeuroVision_MobileNetV2_FT@champion"

IMG_SIZE = (224, 224)

# Must match the exact class order used during training.
CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary",
]


# -----------------------------
# Streamlit page
# -----------------------------

st.set_page_config(
    page_title="NeuroVision AI",
    page_icon="🧠",
    layout="centered",
)

st.title("🧠 NeuroVision AI")
st.subheader("Brain MRI Classification")
st.write(
    "Upload a brain MRI image to obtain a model prediction."
)

st.warning(
    "This is an experimental AI tool, not a medical diagnosis. "
    "Always consult a qualified medical professional."
)


# -----------------------------
# Load the champion model
# -----------------------------

@st.cache_resource
def load_champion_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    return mlflow.tensorflow.load_model(MODEL_URI)


# -----------------------------
# Image upload
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload an MRI image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded MRI",
        use_container_width=True,
    )

    if st.button("Analyze MRI", type="primary"):
        try:
            with st.spinner("Loading model and analyzing image..."):
                model = load_champion_model()

                # Match the input preprocessing used in training.
                resized_image = image.resize(IMG_SIZE)
                image_array = np.asarray(
                    resized_image,
                    dtype=np.float32,
                )

                # Use this ONLY if training used pixel values [0, 1].
                image_array = image_array / 255.0

                # Add batch dimension: (224, 224, 3) -> (1, 224, 224, 3)
                image_array = np.expand_dims(image_array, axis=0)

                predictions = model.predict(
                    image_array,
                    verbose=0,
                )

                probabilities = predictions[0]
                predicted_index = int(np.argmax(probabilities))
                predicted_class = CLASS_NAMES[predicted_index]
                confidence = float(probabilities[predicted_index])

            st.success("Prediction completed")

            st.metric(
                "Predicted class",
                predicted_class.replace("_", " ").title(),
            )

            st.metric(
                "Model score",
                f"{confidence:.2%}",
            )

            st.subheader("Scores by class")

            for class_name, score in zip(
                CLASS_NAMES,
                probabilities,
            ):
                st.write(
                    f"{class_name.title()}: {float(score):.2%}"
                )
                st.progress(
                    float(np.clip(score, 0.0, 1.0))
                )

            st.caption(
                "Scores are model outputs, not guaranteed clinical probabilities."
            )

        except Exception as error:
            st.error(
                "Could not load the model or process the image."
            )
            st.exception(error)
