import tensorflow as tf
import numpy as np
import logging
from models.classifier import NeuralClassifier
from data.data_util import load_data
from training import train
from inference import infer, plot_predictions


logging.basicConfig(level=logging.INFO)
    
logging.info("Loading data...")
x_train, y_train, x_test, y_test = load_data()
logging.info("Creating model...")
model = NeuralClassifier(
    num_hidden_units=3,
    # hidden_layer_activation_function=tf.nn.relu
)

saved_model_dir = train(
    model, x_train, y_train, x_test, y_test,
    model_name="h3",
    plot_every=100
)

plot_predictions(
    model,
    x_train, y_train, x_test, y_test,
    saved_model_dir=saved_model_dir
)
