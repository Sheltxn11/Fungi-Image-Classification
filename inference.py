import tensorflow as tf
import numpy as np
from config import LABELS

def run_inference(model_path, test_data):
    model = tf.keras.models.load_model(model_path)
    
    # Assuming `test_data` is a batch of images
    predictions = model.predict(test_data)
    
    predicted_labels = [LABELS[np.argmax(pred)] for pred in predictions]
    
    print("Predicted Labels:", predicted_labels)
