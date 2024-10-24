import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocess_data(train_path):
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_data = datagen.flow_from_directory(
        train_path,
        target_size=(224, 224),
        batch_size=64,
        class_mode='sparse',
        subset='training'
    )
    
    val_data = datagen.flow_from_directory(
        train_path,
        target_size=(224, 224),
        batch_size=64,
        class_mode='sparse',
        subset='validation'
    )
    
    return train_data, val_data, None  # Modify if test data is needed
