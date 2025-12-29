import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

st.set_page_config(page_title="CNN Eye Detector", page_icon="👁️")
st.title("👁️ MRL Eye State Classifier")

@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('eye_state_cnn.h5')

model = load_my_model()

uploaded_file = st.file_uploader("Upload an eye image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Input Image", width=200)
    
    # Preprocessing to match training data
    image = ImageOps.grayscale(image)
    image = image.resize((64, 64))
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 64, 64, 1) # Add batch and channel dim
    
    # Predict
    prediction = model.predict(img_array)[0][0]
    label = "OPEN" if prediction > 0.5 else "CLOSED"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    
    st.metric(label="Predicted State", value=label)
    st.write(f"Confidence: {confidence:.2%}")
