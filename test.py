import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define dataset and model paths
dataset_path = "dataset"
test_path = os.path.join(dataset_path, "test")
model_path = "models/tumor_classification.h5"

# Check for trained model
if not os.path.exists(model_path):
    raise FileNotFoundError("Model file not found. Train the model first!")

# Load the model
model = tf.keras.models.load_model(model_path)

# Class labels
tumor_classes = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# Load test data
datagen = ImageDataGenerator(rescale=1.0 / 255.0)
test_generator = datagen.flow_from_directory(
    test_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

# Print class mappings
print("\nClass Indices Mapping:", test_generator.class_indices)

# Evaluate model
loss, accuracy = model.evaluate(test_generator)
print(f"\nModel Evaluation:")
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Predictions
y_true = test_generator.classes
y_prob = model.predict(test_generator)
y_pred = np.argmax(y_prob, axis=1)

# ----------- 🔷 CONFUSION MATRIX ------------
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=tumor_classes, yticklabels=tumor_classes)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# ----------- 🔷 CLASSIFICATION REPORT ------------
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=tumor_classes))

# ----------- 🔷 ROC CURVE (Multiclass) ------------
y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3])

plt.figure(figsize=(8, 6))
for i in range(len(tumor_classes)):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{tumor_classes[i]} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.title("Multiclass ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()

# ----------- 🔷 OPTIONAL: Predict Custom Image ------------
def predict_image(img_path):
    if not os.path.exists(img_path):
        print("Image not found.")
        return
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = tumor_classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100
    print(f"\nPrediction for {os.path.basename(img_path)}:")
    print(f"Predicted Class: {predicted_class} (Confidence: {confidence:.2f}%)")

# Change this to your test image path
sample_image_path = "D:/Notes/Degree/Projects/Brain Tumor Detection Using CNN/dataset/train/notumor/Tr-no_0017.jpg"
predict_image(sample_image_path)

print("\n All evaluation steps completed.")
