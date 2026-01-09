from tensorflow.keras.applications import MobileNetV2

def load_model():
    """
    Loads a MobileNetV2 model for brain tumor detection.
    """
    model = MobileNetV2(weights="imagenet", include_top=False)
    return model
