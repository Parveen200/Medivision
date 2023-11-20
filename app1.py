import streamlit as st
import numpy as np
from PIL import Image
from keras.models import load_model
from tensorflow.keras.models import load_model
import tensorflow as tf

# Load the trained models
pneumonia_model = load_model("PneumoniaDetectionModel.h5")
brain_tumor_model = load_model("BrainTumorDetectionModel.h5")
breast_cancer_model = load_model("BreastCancerDetectionModel.h5")

# Configure the page layout
st.set_page_config(
    page_title="Medical Condition Detection",
    page_icon=":microscope:",
    layout="wide"
)

# Create a navigation bar at the top
page= st.sidebar.selectbox("Select a medical condition:", ('Pneumonia Detection', 'Brain Tumor', 'Breast Cancer'))

# Streamlit dashboard
st.title(page)

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg","png"])

if uploaded_file is not None:
    try:
        # Try opening the image
        image = Image.open(uploaded_file)

        # Display the uploaded image with adjusted width
        st.image(image, caption="Uploaded Image", use_column_width=True, width=400)

        # Preprocess the image based on the selected medical condition
        if page == 'Pneumonia Detection':
            # Preprocess the image for the pneumonia model
            image = image.resize((64, 64)).convert("RGB")
            image_array = np.array(image)
            processed_image = image_array.reshape((1, 64, 64, 3))
            processed_image = processed_image.astype('float32') / 255

            # Predict using the pneumonia model
            prediction = pneumonia_model.predict(processed_image)
            result = 'Pneumonia' if prediction[0][0] > 0.5 else 'Normal'

            st.write("Prediction Result:", result)
            st.write("Confidence:", f"{prediction[0][0]*100:.2f}%")

        elif page == 'Brain Tumor':
            # Preprocess the image for the brain tumor model
            image = image.resize((64, 64))
            image_array = np.array(image)
            image_array = image_array.reshape((1, 64, 64, 3))
            image_array = image_array.astype('float32') / 255

            # Predict using the brain tumor model
            prediction = brain_tumor_model.predict(image_array)
            result = 'Brain Tumor Detected' if prediction[0][0] > 0.5 else 'No Brain Tumor Detected'

            st.write("Prediction Result:", result)
            st.write("Confidence:", f"{prediction[0][0]*100:.2f}%")

        elif page == 'Breast Cancer':
            # Preprocess the image for the breast cancer model
            image = image.resize((64, 64))
            image_array = np.array(image)
            image_array = image_array.reshape((1, 64, 64, 3))
            image_array = image_array.astype('float32') / 255

            # Predict using the breast cancer model
            prediction = breast_cancer_model.predict(image_array)
            result = 'Breast Cancer Detected' if prediction[0][0] > 0.5 else 'No Breast Cancer Detected'

            st.write("Prediction Result:", result)
            st.write("Confidence:", f"{prediction[0][0]*100:.2f}%")

    except Exception as e:
        # Display an error message if there's an issue with the image
        st.error(f"Error: {e}")
