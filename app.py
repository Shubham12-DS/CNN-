import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import cv2
import os

# --- Page Configuration ---
st.set_page_config(page_title="AI Eye State Monitor", page_icon="👁️")

st.markdown("""
    <style>
    .main {text-align: center;}
    .stMetric {background-color: #f0f2f6; padding: 10px; border-radius: 10px;}
    </style>
    """, unsafe_allow_status_code=True)

st.title("👁️ Smart Eye State Classifier")
st.info("Upload a photo or use the camera to detect if eyes are Open or Closed.")

# --- Load Model & Face Detector ---
@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model('eye_state_cnn.h5')
    # Use OpenCV's built-in Haar Cascade for eye detection
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    return model, eye_cascade

model, eye_cascade = load_resources()

# --- Input Choice ---
input_mode = st.radio("Select Input:", ["Camera Snapshot", "Upload Image"])

if input_mode == "Camera Snapshot":
    img_file = st.camera_input("Take a photo")
else:
    img_file = st.file_uploader("Choose an image", type=['jpg', 'png', 'jpeg'])

if img_file:
    # Convert to OpenCV format
    image = Image.open(img_file)
    frame = np.array(image)
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Detect Eyes
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(eyes) == 0:
        st.warning("No eyes detected. Please look directly at the camera.")
    else:
        for (x, y, w, h) in eyes[:1]: # Take the first detected eye
            # Extract and Preprocess the eye region
            roi_gray = gray[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi_gray, (64, 64))
            roi_normalized = roi_resized / 255.0
            roi_input = roi_normalized.reshape(1, 64, 64, 1)
            
            # Predict
            prediction = model.predict(roi_input)[0][0]
            label = "OPEN" if prediction > 0.5 else "CLOSED"
            confidence = prediction if prediction > 0.5 else 1 - prediction
            
            # Display Results
            col1, col2 = st.columns(2)
            with col1:
                st.image(roi_gray, caption="Detected Eye Region", width=150)
            with col2:
                color = "green" if label == "OPEN" else "red"
                st.markdown(f"### Result: :{color}[{label}]")
                st.write(f"Confidence: **{confidence:.2%}**")
                
                if label == "CLOSED":
                    st.warning("⚠️ Drowsiness Detected!")
                else:
                    st.success("✅ Driver is Alert")
