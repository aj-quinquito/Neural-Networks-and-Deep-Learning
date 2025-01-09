import streamlit as st
import tensorflow as tf
import numpy as np
import os
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

# Set page layout to wide
st.set_page_config(page_title="Image Classification", layout="wide")

# Path to the dataset directory
dataset_dir = 'C:/Users/AJ/OneDrive/Desktop/school/Neural Networks and Deep Learning/Weeds and Crops Classification/dataset_1'

# Sidebar: Load models
@st.cache_resource
def load_models():
    vgg16_model = tf.keras.models.load_model(
        "C:/Users/AJ/OneDrive/Desktop/school/Neural Networks and Deep Learning/Weeds and Crops Classification/code/vgg16_model.h5")
    MobileNet_model = tf.keras.models.load_model(
        "C:/Users/AJ/OneDrive/Desktop/school/Neural Networks and Deep Learning/Weeds and Crops Classification/code/MobileNet_model.h5")
    simple_cnn_model = tf.keras.models.load_model(
        "C:/Users/AJ/OneDrive/Desktop/school/Neural Networks and Deep Learning/Weeds and Crops Classification/code/simple_cnn_model.h5")
    return vgg16_model, MobileNet_model, simple_cnn_model

vgg16_model, MobileNet_model, simple_cnn_model = load_models()

# Get class labels dynamically
class_labels = {i: folder for i, folder in enumerate(sorted(os.listdir(dataset_dir)))}


def preprocess_image(image):
    """Preprocess the image dynamically for prediction."""
    # Resize the image to (128, 128) for all models
    image = image.resize((128, 128))
    image = img_to_array(image)
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


# Sidebar for input controls
st.sidebar.title("Input")
model_option = st.sidebar.selectbox(
    "Choose a model for prediction",
    ("VGG16", "MobileNet", "Simple CNN", "ViT")
)
uploaded_files = st.sidebar.file_uploader(
    "Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)

# Main Title
st.title("Image Classification")
st.markdown("Upload multiple images, select a model, and view predictions dynamically.")

# Reset session state if all files are deleted
if uploaded_files is None or len(uploaded_files) == 0:
    st.session_state["uploaded_images"] = []
    st.session_state["predictions"] = {}

# Store uploaded files
if "uploaded_images" not in st.session_state:
    st.session_state["uploaded_images"] = []
    st.session_state["predictions"] = {}

# Add newly uploaded files to session state
if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state["uploaded_images"]:
            st.session_state["uploaded_images"].append(uploaded_file.name)
            # Process the image for predictions using all models
            image = Image.open(uploaded_file)
            preprocessed_image = preprocess_image(image, model_option if model_option == "ViT" else "default")
            st.session_state["predictions"][uploaded_file.name] = {
                "VGG16": vgg16_model.predict(preprocessed_image),
                "MobileNet": MobileNet_model.predict(preprocessed_image),
                "Simple CNN": simple_cnn_model.predict(preprocessed_image),
                "ViT": vit_model(preprocessed_image).logits.detach().numpy()  # Adjusted for ViT
            }

# Display uploaded images in columns
if st.session_state["uploaded_images"]:
    st.subheader("Uploaded Images and Predictions")
    num_columns = 3
    columns = st.columns(num_columns)

    for i, file_name in enumerate(st.session_state["uploaded_images"]):
        col = columns[i % num_columns]
        with col:
            # Display the image
            for uploaded_file in uploaded_files:
                if uploaded_file.name == file_name:
                    image = Image.open(uploaded_file)
                    st.image(image, caption=f"Image {i + 1}", use_container_width=True)

                    # Get predictions
                    predictions = st.session_state["predictions"][file_name][model_option]
                    predicted_class_index = np.argmax(predictions, axis=1)[0]
                    confidence_score = predictions[0][predicted_class_index]

                    # Display results
                    st.markdown(f"**Model:** {model_option}")
                    st.markdown(f"- **Predicted Class:** {class_labels.get(predicted_class_index, 'Unknown Class')}")
                    st.markdown(f"- **Confidence:** {confidence_score * 100:.2f}%")
else:
    st.info("Upload images to get started.")
