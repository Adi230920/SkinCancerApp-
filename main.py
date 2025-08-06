# main.py

import streamlit as st
from PIL import Image
from utils import load_trained_model, load_label_encoder, preprocess_image

# Load model and encoder once
model = load_trained_model()
label_encoder = load_label_encoder()

st.title("Skin Lesion Type Detection")

uploaded_file = st.file_uploader("Upload a skin lesion image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)


    # Preprocess
    processed = preprocess_image(image)

    # Predict
    prediction = model.predict(processed)
    predicted_class = label_encoder.inverse_transform([prediction.argmax()])[0]

    st.success(f"Predicted Lesion Type: **{predicted_class}**")





