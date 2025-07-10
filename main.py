import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, render_template
import grad_cam  # Import updated Grad-CAM script
from tensorflow.keras.applications import MobileNetV2

# Initialize Flask app
app = Flask(__name__)

# Define paths
MODEL_TUMOR_DETECTION = "models/tumor_detection.h5"
MODEL_TUMOR_CLASSIFICATION = "models/tumor_classification.h5"
UPLOAD_FOLDER = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load models
try:
    # Load the tumor detection model
    tumor_detection_model = load_model(MODEL_TUMOR_DETECTION)

    # Load your trained tumor classification model
    tumor_classification_model = load_model(MODEL_TUMOR_CLASSIFICATION)

    # Load MobileNetV2 for Grad-CAM (temporary)
    gradcam_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

    # Initialize models with dummy input
    dummy_input = np.zeros((1, 150, 150, 3), dtype=np.float32)
    tumor_detection_model.predict(dummy_input)
    tumor_classification_model.predict(dummy_input)
    gradcam_model.predict(dummy_input)

    print("Models loaded and initialized successfully!")
    print(f"[DEBUG main.py] Model Name after loading: {tumor_classification_model.name}")
except Exception as e:
    print(f"Error loading models: {e}")
    exit()

# Function to check allowed file extensions
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess image
def preprocess_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(150, 150))
        img_array = image.img_to_array(img) / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

@app.route("/", methods=["GET", "POST"])
def index():
    gradcam_path = None
    if request.method == "POST":
        file = request.files.get("file")

        if not file or file.filename == "":
            return render_template("index.html", error="No file uploaded")

        if file and allowed_file(file.filename):
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # Preprocess the image
            img_array = preprocess_image(filepath)
            if img_array is None:
                return render_template("index.html", error="Invalid image format")

            # Step 1: Tumor Detection
            tumor_pred = tumor_detection_model.predict(img_array)[0][0]
            if tumor_pred < 0.5:
                return render_template("index.html", result="No Tumor Detected", image_path=filepath)

            # Step 2: Tumor Type Classification
            tumor_class_pred = tumor_classification_model.predict(img_array)
            class_labels = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
            predicted_class_index = np.argmax(tumor_class_pred)
            predicted_class = class_labels[predicted_class_index]
            confidence = tumor_class_pred[0][predicted_class_index] * 100

            # Generate Grad-CAM visualization (using MobileNetV2)
            gradcam_result = grad_cam.generate_gradcam(filepath, gradcam_model)
            if gradcam_result:
                gradcam_path, _ = gradcam_result
            else:
                gradcam_path = None

            return render_template(
                "index.html",
                result=f"{predicted_class} ({confidence:.2f}% Confidence)",
                image_path=filepath,
                gradcam_path=gradcam_path
            )

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)