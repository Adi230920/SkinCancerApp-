# utils.py

import numpy as np
from PIL import Image
import joblib
from tensorflow.keras.models import load_model

SIZE = 32

def load_trained_model(path='model/skin_cancer_model.h5'):
    return load_model(path)

def load_label_encoder(path='model/label_encoder.pkl'):
    return joblib.load(open(path, 'rb'))

def preprocess_image(image):
    # image is a PIL Image object already
    img = image.convert("RGB").resize((SIZE, SIZE))  # Ensure 3-channel RGB and resize
    img_array = np.array(img) / 255.0  # Normalize
    img_array = img_array.reshape(1, SIZE, SIZE, 3)  # Reshape for model input
    return img_array

