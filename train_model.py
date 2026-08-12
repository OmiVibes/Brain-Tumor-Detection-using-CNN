import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight

# Define dataset paths
dataset_path = "dataset"
train_path = os.path.join(dataset_path, "train")
test_path = os.path.join(dataset_path, "test")

# Image augmentation and preprocessing
datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# Ensure models directory exists
os.makedirs("models", exist_ok=True)

# Define model paths
model_tumor_path = "models/tumor_detection.h5"
model_classification_path = "models/tumor_classification.h5"

# Compute class weights
train_generator_full = datagen.flow_from_directory(
    train_path, target_size=(150, 150), batch_size=32, class_mode='categorical', subset="training")

labels = train_generator_full.classes
class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Callbacks for training
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint_tumor = ModelCheckpoint(model_tumor_path, save_best_only=True, monitor='val_loss')
model_checkpoint_classification = ModelCheckpoint(model_classification_path, save_best_only=True, monitor='val_loss')

# Step 1: Train Tumor Detection Model (Binary Classification)
if not os.path.exists(model_tumor_path):
    print("Training Tumor Detection Model (No Tumor vs Tumor)")

    train_generator_binary = datagen.flow_from_directory(
        train_path, target_size=(150, 150), batch_size=32, class_mode='binary', subset="training")

    val_generator_binary = datagen.flow_from_directory(
        train_path, target_size=(150, 150), batch_size=32, class_mode='binary', subset="validation")

    tumor_detection_model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    tumor_detection_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    tumor_detection_model.fit(train_generator_binary, validation_data=val_generator_binary, epochs=20,
                              callbacks=[early_stopping, model_checkpoint_tumor])

    print(f"Tumor Detection Model saved at {model_tumor_path}")
else:
    print("Tumor Detection Model already trained.")

# Step 2: Train Tumor Type Classification Model (Multi-Class)
if not os.path.exists(model_classification_path):
    print("\nTraining Tumor Type Classification Model (Glioma, Meningioma, Pituitary)")

    train_generator_multi = datagen.flow_from_directory(
        train_path, target_size=(150, 150), batch_size=32, class_mode='categorical', subset="training")

    val_generator_multi = datagen.flow_from_directory(
        train_path, target_size=(150, 150), batch_size=32, class_mode='categorical', subset="validation")

    tumor_classification_model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(4, activation='softmax')
    ])

    tumor_classification_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    tumor_classification_model.fit(train_generator_multi, validation_data=val_generator_multi, epochs=20,
                                   class_weight=class_weights_dict, callbacks=[early_stopping, model_checkpoint_classification])

    print(f"Tumor Classification Model saved at {model_classification_path}")
else:
    print("Tumor Classification Model already trained.")

print("\nTraining Complete! Models are now ready for Multi-Stage Classification.")