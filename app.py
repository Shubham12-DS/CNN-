import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import cv2

# --- PAGE SETUP ---
st.set_page_config(page_title="Real-time Eye Monitor", page_icon="👁️")
st.title("👁️ AI Eye State Detector")
st.write("Use your camera to check if your eyes are open or closed.")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('eye_state_cnn.h5')

model = load_model()

# --- CAMERA INPUT ---
img_file_buffer = st.camera_input("Take a photo of your eye")

if img_file_buffer is not None:
    # 1. Convert buffer to PIL Image
    img = Image.open(img_file_buffer)
    
    # 2. Preprocess (Grayscale, Resize, Normalize)
    # MRL dataset was trained on Grayscale 64x64 images
    img_gray = ImageOps.grayscale(img)
    img_resized = img_gray.resize((64, 64))
    img_array = np.array(img_resized) / 255.0
    img_array = img_array.reshape(1, 64, 64, 1) # Match CNN input shape

    # 3. Prediction
    prediction = model.predict(img_array)[0][0]
    
    # --- UI DISPLAY ---
    if prediction > 0.5:
        st.success("STATE: OPEN")
        st.balloons()
    else:
        st.error("STATE: CLOSED")
        st.warning("Drowsiness Alert!")

    st.write(f"**Confidence Score:** {prediction if prediction > 0.5 else 1 - prediction:.2%}")
