import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image
from edgeDetection import sobel_filter, canny_filter, laplace_filter

# Page styling and title
st.set_page_config(layout="wide")
st.title("TensorFlow U-Net Model")
st.write("Neural Network model using Keras, U-NET, and images from the Broad Bioimage Benchmark Collection")
st.info("The model is working decently well, but still doesn't know how to count cells yet")

# Show a message while loading the model
with st.spinner("Booting Up Neural Network..."):
    # Load the trained U-Net model
    model = load_model(r'C:\Users\chans\Documents\GitHub\CellAnalyzer\unet_model_final.keras', compile=False)

# Helper functions
def preprocess_image(image):
    #image = cv2.bitwise_not(image)
    image = apply_sobel_filter(image)
    #image = apply_laplace_filter(image)
    image = img_to_array(image)
    image = cv2.resize(image, (256, 256))
    image = image / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def predict(image):
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    return (prediction[0] > 0.5).astype(np.uint8)  # Binarize prediction

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "bmp", "tif"])

if uploaded_file is not None:
    # turns the image file into an array that OpenCV can understand and decode
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    # Decode the byte array to an image
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    # Make prediction
    mask = predict(image)

    # Convert mask to an image
    mask_image = (mask.squeeze() * 255).astype(np.uint8)
    mask_image = Image.fromarray(mask_image)

    # Optionally, display the original and mask side by side
    st.write("### Original Image and Mask")
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption='Original Image', use_column_width=True)
    with col2:
        st.image(mask_image, caption='Model Prediction Mask', use_column_width=True, channels="GRAY")
