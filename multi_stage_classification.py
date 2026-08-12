import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Define model paths
MODEL_TUMOR_DETECTION = "models/tumor_detection.h5"
MODEL_TUMOR_CLASSIFICATION = "models/tumor_classification.h5"

# Load models
tumor_detection_model = load_model(MODEL_TUMOR_DETECTION)
tumor_classification_model = load_model(MODEL_TUMOR_CLASSIFICATION)

# Define class labels
class_labels = ["Glioma", "Meningioma", "No Tumor","Pituitary"]

# Function to preprocess image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150), color_mode="rgb")
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Expand to batch dimension
    return img_array

# Function for multi-stage classification
def classify_brain_tumor(img_path):
    img_array = preprocess_image(img_path)

    # Step 1: Tumor Detection
    tumor_pred = tumor_detection_model.predict(img_array)[0][0]

    if tumor_pred < 0.5:
        return "No Tumor Detected", None, None

    # Step 2: Tumor Classification
    tumor_class_pred = tumor_classification_model.predict(img_array)[0]
    predicted_index = np.argmax(tumor_class_pred)
    predicted_class = class_labels[predicted_index]
    confidence = np.max(tumor_class_pred) * 100

    return predicted_class, confidence, predicted_index

# Test function
if __name__ == "__main__":
    test_image = "D:/Notes/Degree/Projects/Brain Tumor Detection Using CNN/static/uploads/Te-gl_0010.jpg"  # Replace with an actual test image
    result, confidence, index = classify_brain_tumor(test_image)
    if result == "No Tumor Detected":
        print(result)
    else:
        print(f"Predicted: {result}, Confidence: {confidence:.2f}%")
