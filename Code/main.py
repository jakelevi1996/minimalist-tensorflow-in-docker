import tensorflow as tf
import numpy as np
import logging
from models.classifier import NeuralClassifier
from data.data_util import load_data, DEFAULT_FILENAME
from training import train
from inference import infer, plot_predictions


logging.basicConfig(level=logging.INFO)
    
logging.info("Loading data...")
x_train, y_train, x_test, y_test = load_data(DEFAULT_FILENAME)
logging.info("Creating model...")
model = NeuralClassifier()

saved_model_dir = train(
    model, x_train, y_train, x_test, y_test,
    # saved_model_dir="models/benchmark/",
    # num_epochs=3
)

y_pred = infer(model, saved_model_dir, x_test)
plot_predictions(x_train, y_train, x_test, y_pred)