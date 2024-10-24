import tensorflow as tf
from config import EPOCHS, MODEL_PATH
from callbacks import get_callbacks

def build_model():
    base_model = tf.keras.applications.ResNet50V2(input_shape=(224, 224, 3), include_top=False)
    base_model.trainable = False
    
    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(5, activation='softmax', kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01))
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

def train_model(train_data, val_data):
    model = build_model()
    
    callbacks = get_callbacks()
    
    model.fit(train_data, validation_data=val_data, epochs=EPOCHS, callbacks=callbacks)
    
    model.save(MODEL_PATH)
    
    return model
