import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.set_page_config(
    page_title="Tomato Disease Detection",
    page_icon="🌱",
    layout="centered"
)

st.title("🌱 Tomato Disease Detection System")
st.write("Upload a tomato leaf image to detect whether it is healthy or diseased.")

MODEL_PATH = "crop_disease_model.h5"
LABELS_PATH = "labels.txt"

if not os.path.exists(MODEL_PATH):
    st.error("Model file 'crop_disease_model.h5' not found.")
    st.stop()

if not os.path.exists(LABELS_PATH):
    st.error("Labels file 'labels.txt' not found.")
    st.stop()

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_data
def load_labels():
    with open(LABELS_PATH, "r") as f:
        return [line.strip() for line in f.readlines()]

model = load_model()
labels = load_labels()

remedies = {
    "Tomato_Early_Blight": "Remove infected leaves and apply fungicide.",
    "Tomato_Late_Blight": "Use copper-based spray and avoid excess moisture.",
    "Tomato_Healthy": "Plant appears healthy. Continue proper care."
}

CONFIDENCE_THRESHOLD = 0.75

uploaded_file = st.file_uploader(
    "Upload tomato leaf image",
    type=["jpg", "jpeg", "png"]
)

def predict_image(image: Image.Image):
    image = image.convert("RGB")
    resized = image.resize((224, 224))

    # IMPORTANT: no /255 here
    img_array = np.array(resized, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)[0]
    predicted_index = int(np.argmax(prediction))
    confidence = float(prediction[predicted_index])
    predicted_label = labels[predicted_index]

    return predicted_label, confidence, prediction

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing leaf image..."):
        predicted_label, confidence, all_scores = predict_image(image)

    st.subheader("Prediction Result")
    st.write(f"**Predicted Class:** {predicted_label}")
    st.write(f"**Confidence:** {confidence:.2%}")

    if confidence < CONFIDENCE_THRESHOLD:
        st.warning("This result has low confidence. Please upload a clearer tomato leaf image.")
    else:
        st.success("Prediction confidence is acceptable.")

    st.subheader("Class Scores")
    for i, label in enumerate(labels):
        st.write(f"**{label}:** {all_scores[i]:.4f}")

    st.subheader("Suggested Remedy")
    st.write(remedies.get(predicted_label, "No remedy available."))

else:
    st.info("Please upload a tomato leaf image to begin.")