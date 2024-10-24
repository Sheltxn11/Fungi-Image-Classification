from data_preprocessing import preprocess_data
from model_training import train_model
from inference import run_inference
from config import TRAIN_PATH, MODEL_PATH

if __name__ == "__main__":

    train_data, val_data, test_data = preprocess_data(TRAIN_PATH)
    model = train_model(train_data, val_data)
    run_inference(MODEL_PATH, test_data)
