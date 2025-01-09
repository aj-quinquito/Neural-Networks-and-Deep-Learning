import streamlit as st
import tensorflow as tf
import numpy as np
import os
import cv2
import json
from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import torch

# Set page layout to wide
st.set_page_config(page_title="Image Classification", layout="wide")

# Paths
dataset_dir = 'C:/Users/AJ/OneDrive/Desktop/school/Neural Networks and Deep Learning/Weeds and Crops Classification/dataset_1/'
model_metadata_paths = {
    "VGG16": "C:/Users/AJ/OneDrive/Desktop/school/Neural Networks and Deep Learning/Weeds and Crops Classification/code/training codes/vgg16_model_metadata.json",
    "MobileNet": "C:/Users/AJ/OneDrive/Desktop/school/Neural Networks and Deep Learning/Weeds and Crops Classification/code/training codes/MobileNet_metadata.json",
    "Simple CNN": "C:/Users/AJ/OneDrive/Desktop/school/Neural Networks and Deep Learning/Weeds and Crops Classification/code/training codes/simple_cnn_metadata.json",
    "Segmented (VGG16)": "C:/Users/AJ/OneDrive/Desktop/school/Neural Networks and Deep Learning/Weeds and Crops Classification/code/training codes/pt_model_4.json"
}

# -------------------------------------------------------------------------------------------------------------------------
# Helper function for segmentation
def segment_image(image):
    """
    Segment the input image using HSV-based segmentation.
    """
    # Convert PIL image to numpy array
    image_np = np.array(image)

    # Apply Gaussian blur and convert to HSV
    blurr = cv2.GaussianBlur(image_np, (5, 5), 0)
    hsv = cv2.cvtColor(blurr, cv2.COLOR_RGB2HSV)  # Convert to HSV (RGB in PIL)

    # Define green color range for segmentation
    sensitivity = 30
    lower_hsv = np.array([60 - sensitivity, 100, 50])
    upper_hsv = np.array([60 + sensitivity, 255, 255])

    # Create mask and apply morphological operations
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Create segmented image
    masked = mask > 0
    preprocessed = np.zeros_like(image_np, np.uint8)
    preprocessed[masked] = image_np[masked]

    # Convert back to PIL Image
    segmented_image = Image.fromarray(preprocessed)
    return segmented_image

# -------------------------------------------------------------------------------------------------------------------------

# Sidebar: Load models and metadata
@st.cache_resource
def load_models_and_metadata():
    # Load TensorFlow models
    vgg16_model = tf.keras.models.load_model("C:/Users/AJ/OneDrive/Desktop/school/Neural Networks and Deep Learning/Weeds and Crops Classification/code/training codes/vgg16_model.h5")
    MobileNet_model = tf.keras.models.load_model("C:/Users/AJ/OneDrive/Desktop/school/Neural Networks and Deep Learning/Weeds and Crops Classification/code/training codes/MobileNet_model.h5")
    simple_cnn_model = tf.keras.models.load_model("C:/Users/AJ/OneDrive/Desktop/school/Neural Networks and Deep Learning/Weeds and Crops Classification/code/training codes/simple_cnn_model.h5")
    segmented_vgg16 = tf.keras.models.load_model("C:/Users/AJ/OneDrive/Desktop/school/Neural Networks and Deep Learning/Weeds and Crops Classification/code/training codes/pt_model_4.h5")
    
    # Load metadata
    vgg16_metadata = json.load(open(model_metadata_paths["VGG16"]))
    mobilenet_metadata = json.load(open(model_metadata_paths["MobileNet"]))
    simple_cnn_metadata = json.load(open(model_metadata_paths["Simple CNN"]))
    segmented_vgg16_metadata = json.load(open(model_metadata_paths["Segmented (VGG16)"]))
    
    return (vgg16_model, vgg16_metadata, 
            MobileNet_model, mobilenet_metadata, 
            simple_cnn_model, simple_cnn_metadata, 
            segmented_vgg16, segmented_vgg16_metadata)

# Load models and metadata
vgg16_model, vgg16_metadata, MobileNet_model, mobilenet_metadata, simple_cnn_model, simple_cnn_metadata,  segmented_vgg16, segmented_vgg16_metadata = load_models_and_metadata()

# Preprocess images dynamically
def preprocess_image(image, model_option):
    if model_option in ["VGG16", "MobileNet", "Simple CNN", "Segmented (VGG16)"]:
        image = image.resize((128, 128))
        image = img_to_array(image)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)

    return image

# -------------------------------------------------------------------------------------------------------------------------

# Sidebar inputs
st.sidebar.title("Input")
model_option = st.sidebar.selectbox("Choose a model for prediction", ("VGG16", "MobileNet", "Simple CNN", "Segmented (VGG16)"))

# Retrieve and display validation accuracy dynamically
if model_option == "VGG16":
    val_accuracy = vgg16_metadata["val_accuracy"]
elif model_option == "MobileNet":
    val_accuracy = mobilenet_metadata["val_accuracy"]
elif model_option == "Simple CNN":
    val_accuracy = simple_cnn_metadata["val_accuracy"]
elif model_option == "Segmented (VGG16)":
    val_accuracy = segmented_vgg16_metadata["val_accuracy"]
else:
    val_accuracy = None

# Display validation accuracy under the dropdown
if val_accuracy is not None:
    st.sidebar.markdown(f"**Validation Accuracy:** {val_accuracy * 100:.2f}%")
else:
    st.sidebar.markdown("**Validation Accuracy:** Not Available")

uploaded_files = st.sidebar.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# -------------------------------------------------------------------------------------------------------------------------

# Track the selected model in session state
if "selected_model" not in st.session_state:
    st.session_state["selected_model"] = model_option

# Check if model selection has changed
if st.session_state["selected_model"] != model_option:
    st.session_state["selected_model"] = model_option
    st.session_state["uploaded_images"] = []
    st.session_state["predictions"] = {}
    st.session_state["true_labels"] = {}

# -------------------------------------------------------------------------------------------------------------------------

# Main Title
st.title("Weeds and Crops Image Classification")
st.markdown("Upload images, select a model, and view predictions.")

# Store uploaded files
if "uploaded_images" not in st.session_state:
    st.session_state["uploaded_images"] = []
    st.session_state["predictions"] = {}

# -------------------------------------------------------------------------------------------------------------------------

def get_true_label(dataset_dir):
    image_label_mapping = {}
    
    for subdir in os.scandir(dataset_dir):
        if subdir.is_dir():
            label = subdir.name  # The folder name is the label
            for file in os.scandir(subdir.path):
                if file.is_file():
                    image_label_mapping[os.path.basename(file.path)] = label
    
    return image_label_mapping

image_label_mapping = get_true_label(dataset_dir)

# -------------------------------------------------------------------------------------------------------------------------

def temperature_scaled_softmax(logits, temperature=1.0):
    """
    Apply temperature scaling to logits before softmax.
    """
    scaled_logits = logits / temperature
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))  # Stability fix
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

# -------------------------------------------------------------------------------------------------------------------------

label_normalization = {
    "Shepherd’s Purse": "Shepherds Purse",
    "Shepherds Purse": "Shepherds Purse",
    "ShepherdÕÇÕs Purse": "Shepherds Purse",
    "ShepherdΓÇÖs Purse": "Shepherds Purse",
}

def normalize_label(label):
    """
    Normalize a label to the standard naming convention.
    """
    normalized_label = label_normalization.get(label.strip(), label.strip())
    return normalized_label


# -------------------------------------------------------------------------------------------------------------------------

# Process uploaded files
if uploaded_files:
    current_uploaded_files = [file.name for file in uploaded_files]
    
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state["uploaded_images"]:
            st.session_state["uploaded_images"].append(uploaded_file.name)
            
            # Preprocess and predict
            image = Image.open(uploaded_file).convert("RGB")
            
            # Apply segmentation if "Segmented (VGG16)" is selected
            if model_option == "Segmented (VGG16)":
                image = segment_image(image)  # Segment the image dynamically
            
            # Preprocess the image for the selected model
            preprocessed_image = preprocess_image(image, model_option)
            
            if model_option == "VGG16":
                preds = vgg16_model.predict(preprocessed_image)
                val_accuracy = vgg16_metadata["val_accuracy"]
                class_indices = vgg16_metadata.get("class_indices", {})
            elif model_option == "MobileNet":
                preds = MobileNet_model.predict(preprocessed_image)
                val_accuracy = mobilenet_metadata["val_accuracy"]
                class_indices = mobilenet_metadata.get("class_indices", {})
            elif model_option == "Simple CNN":
                preds = simple_cnn_model.predict(preprocessed_image)
                val_accuracy = simple_cnn_metadata["val_accuracy"]
                class_indices = simple_cnn_metadata.get("class_indices", {})
            elif model_option == "Segmented (VGG16)":
                preds = segmented_vgg16.predict(preprocessed_image)
                val_accuracy = segmented_vgg16_metadata["val_accuracy"]
                class_indices = segmented_vgg16_metadata.get("class_indices", {})


            if model_option in ["VGG16", "MobileNet", "Simple CNN", "Segmented (VGG16)"]:
                logits = preds  # Assuming preds are logits directly from the model
                calibrated_probs = temperature_scaled_softmax(logits, temperature=1.5)
                predicted_class_index = np.argmax(calibrated_probs, axis=1)[0]
                confidence_score = calibrated_probs[0][predicted_class_index] * 100

            predicted_class_index = np.argmax(preds, axis=1)[0]
            confidence_score = preds[0][predicted_class_index]
            
            # Retrieve the true label
            file_name = os.path.basename(uploaded_file.name)
            true_class_name = image_label_mapping.get(file_name, "Unknown")
            
            # Get Predicted and True Class Names
            if class_indices:
                index_to_class = {v: k for k, v in class_indices.items()}
                predicted_class_name = index_to_class.get(predicted_class_index, "Unknown Class")
            else:
                predicted_class_name = "Not Available"

            # Normalize the predicted and true labels
            normalized_predicted_class = normalize_label(predicted_class_name)
            normalized_true_label = normalize_label(true_class_name)

            # Update `result_status` based on normalized labels
            result_status = "Correct" if normalized_predicted_class.strip().lower() == normalized_true_label.strip().lower() else "Misclassified"

            # Save predictions
            st.session_state["predictions"][uploaded_file.name] = {
                "Predicted Class": normalized_predicted_class,  # Use normalized labels
                "Confidence": confidence_score,
                "Validation Accuracy": val_accuracy,
                "True Label": normalized_true_label,  # Use normalized labels
                "Result Status": result_status
            }

    # Remove deleted files
    st.session_state["uploaded_images"] = [
        name for name in st.session_state["uploaded_images"] if name in current_uploaded_files
    ]
    # Also clear predictions for removed files
    st.session_state["predictions"] = {
        name: st.session_state["predictions"][name]
        for name in st.session_state["uploaded_images"]
    }

# -------------------------------------------------------------------------------------------------------------------------

# Define a function to resize images
def resize_image(image, target_size=(200, 200)):
    return image.resize(target_size, Image.ANTIALIAS)

# -------------------------------------------------------------------------------------------------------------------------  

# Display predictions dynamically
if st.session_state["uploaded_images"]:
    st.subheader("Uploaded Images and Predictions")
    columns = st.columns(3)  # Three columns for better organization

    for i, file_name in enumerate(st.session_state["uploaded_images"]):  # Loop through uploaded images
        with columns[i % 3]:  # Place images and details in alternating columns
            file_path = next((f for f in uploaded_files if f.name == file_name), None)
            if file_path:
                # Load the original image
                original_image = Image.open(file_path).convert("RGB")

                # Segment the image if "Segmented (VGG16)" is selected
                if model_option == "Segmented (VGG16)":
                    segmented_image = segment_image(original_image)
                    original_image_resized = original_image.resize((300, 300))
                    segmented_image_resized = segmented_image.resize((300, 300))
                    st.image(original_image_resized, caption=f"Original Image {i + 1}", use_container_width=True)
                    st.image(segmented_image_resized, caption=f"Segmented Image {i + 1}", use_container_width=True)
                else:
                    original_image_resized = original_image.resize((300, 300))
                    st.image(original_image_resized, caption=f"Uploaded Image {i + 1}", use_container_width=True)

                # Display predictions
                predictions = st.session_state["predictions"].get(file_name, {})
                confidence_score = predictions.get('Confidence', 0) * 100  # Retrieve confidence score
                st.markdown(f"**Model:** {model_option}")
                st.markdown(f"- **Index:** {i + 1}")
                st.markdown(f"- **Predicted Class:** {predictions.get('Predicted Class', 'N/A')}")
                st.markdown(f"- **True Label:** {predictions.get('True Label', 'N/A')}")
                st.markdown(f"- **Confidence:** {confidence_score:.2f}%")

                # Highlight misclassified samples and handle correctly predicted cases
                if predictions.get("Result Status") == "Misclassified":
                    if confidence_score > 70.0:  # Threshold for overconfidence
                        st.error("⚠️ This sample was **MISCLASSIFIED** with high confidence!")
                        prediction_category = "Overconfident"
                    else:
                        st.warning("⚠️ This sample was **MISCLASSIFIED** with low confidence!")
                        prediction_category = "Uncertain"
                elif predictions.get("Result Status") == "Correct":
                    if confidence_score >= 70.0:  # Threshold for confident predictions
                        st.success("✅ This sample was **CORRECTLY** predicted with high confidence!")
                        prediction_category = "Confident"
                    else:
                        st.warning("⚠️ This sample was **CORRECTLY** predicted but with low confidence!")
                        prediction_category = "Uncertain"
                else:
                    prediction_category = "Unknown"
else:
    st.info("Upload images to get started.")

