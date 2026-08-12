import numpy as np
import tensorflow as tf
import cv2
import os

def generate_gradcam(image_path, model):
    try:
        print("\n[INFO] Step 1: Loading and Preprocessing Image...")
        img = tf.keras.preprocessing.image.load_img(image_path, target_size=(150, 150))
        img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 150, 150, 3)
        print(f"[DEBUG] Image shape after processing: {img_array.shape}")

        # Ensure the model is called with dummy input to initialize it
        dummy_input = np.zeros((1, 150, 150, 3), dtype=np.float32)
        try:
            model.predict(dummy_input)  # Force the model to initialize.
            print("[DEBUG] Model initialized successfully with dummy input.")
        except Exception as init_error:
            print(f"[ERROR] Model initialization failed: {init_error}")
            return None, None

        print("[INFO] Step 2: Getting Predictions...")
        predictions = model(img_array, training=False)  # Ensure model is properly called

        # Handle MobileNetV2 output: take the mean of the feature map
        confidence_score = np.mean(predictions)
        print(f"[DEBUG] Confidence: {confidence_score:.4f}")

        print("[INFO] Step 3: Identifying Last Convolutional Layer...")
        last_conv_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer.name
                break

        if last_conv_layer is None:
            raise ValueError("No convolutional layer found in the model.")

        print(f"[DEBUG] Last Conv Layer Found: {last_conv_layer}")

        # debugging prints
        print(f"[DEBUG] Model Layers: {model.layers}")
        print(f"[DEBUG] Model Inputs: {model.inputs}")
        print(f"[DEBUG] Model Name: {model.name}")
        print(f"[DEBUG] get_layer output: {model.get_layer(last_conv_layer)}")  # added line.

        print("[INFO] Step 4: Creating Grad-CAM Model Using Functional API...")
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer).output, model.output] #changed line
        )

        print("[INFO] Step 5: Running Grad-CAM Forward Pass...")
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model([img_array], training=False)
            loss = tf.reduce_mean(predictions) #changed line

        grads = tape.gradient(loss, conv_outputs)
        if grads is None:
            raise ValueError("[ERROR] Gradient computation failed. Check model architecture.")

        print(f"[DEBUG] Grads Shape: {grads.shape}")  # added line

        print("[INFO] Step 6: Pooling Gradients...")
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        print("[INFO] Step 7: Applying Gradients to Feature Maps...")
        conv_outputs = conv_outputs[0]  # Remove batch dimension
        heatmap = np.sum(conv_outputs * pooled_grads, axis=-1)
        heatmap = np.maximum(heatmap, 0)  # Apply ReLU activation
        heatmap /= np.max(heatmap) if np.max(heatmap) != 0 else 1  # Normalize

        print("[INFO] Step 8: Loading Original Image...")
        img = cv2.imread(image_path)
        img = cv2.resize(img, (150, 150))

        print("[INFO] Step 9: Resizing Heatmap...")
        heatmap = cv2.resize(heatmap, (150, 150))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        print("[INFO] Step 10: Superimposing Heatmap on Image...")
        superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

        print("[INFO] Step 11: Saving Grad-CAM Image...")
        gradcam_dir = "static/gradcam/"
        os.makedirs(gradcam_dir, exist_ok=True)

        image_ext = os.path.splitext(image_path)[-1].lower()
        gradcam_filename = os.path.basename(image_path).replace(image_ext, "_gradcam.jpg")
        gradcam_path = os.path.join(gradcam_dir, gradcam_filename)

        cv2.imwrite(gradcam_path, superimposed_img)
        print(f"[SUCCESS] Grad-CAM Image Saved at: {gradcam_path}")

        return gradcam_path, confidence_score  # Return path and confidence for Flask UI

    except Exception as e:
        print(f"[ERROR] Grad-CAM Error: {e}")

        # Debugging: Print Model Layers
        print("\n[DEBUG] Model Layers:")
        for i, layer in enumerate(model.layers):
            print(f"  [{i}] {layer.name} - {type(layer)}")

        return None, None