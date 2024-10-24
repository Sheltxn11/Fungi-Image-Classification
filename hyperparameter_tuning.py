from kerastuner.tuners import RandomSearch
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.optimizers import Adam
from data_preprocessing import load_and_preprocess_data
from config import TRAIN_PATH

def build_tuned_model(hp):
    base_model = ResNet50V2(input_shape=(224, 224, 3), include_top=False)
    base_model.trainable = False
    
    model = Sequential([
        base_model,
        Flatten(),
        Dropout(hp.Float('dropout', 0.5, 0.9, step=0.1)),
        Dense(5, activation='softmax', kernel_regularizer=l1_l2(l1=0.01, l2=0.01))
    ])
    
    model.compile(optimizer=Adam(hp.Choice('learning_rate', [1e-3, 1e-4])),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    train_data, val_data = load_and_preprocess_data(TRAIN_PATH)

    tuner = RandomSearch(
        build_tuned_model,
        objective='val_accuracy',
        max_trials=5,
        executions_per_trial=3,
        directory='logs',
        project_name='hyperparameter_tuning'
    )

    tuner.search(train_data, epochs=10, validation_data=val_data)
    tuner.results_summary()
