import tensorflow as tf

# Load model
model_path = "models/tumor_classification.h5"  # Ensure you're using the correct model
model = tf.keras.models.load_model(model_path)

# Print model summary
model.summary()

# Find the last convolutional layer
for layer in model.layers:
    if "conv" in layer.name:
        last_conv_layer = layer.name
print(f"Last Conv Layer: {last_conv_layer}")
