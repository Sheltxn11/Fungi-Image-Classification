# callbacks.py
import tensorflow as tf
from config import MODEL_PATH

def get_callbacks():
    early_stopping = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True)
    
    return [early_stopping, model_checkpoint]
