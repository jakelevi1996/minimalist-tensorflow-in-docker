import numpy as np
import time
import logging

DEFAULT_FILENAME = "data/synthetic_data.npz"

def classification_rule(x):
    bool_val = (x[:,0]**2 + x[:,1]**2) < 1
    y = (1 * bool_val).reshape(-1,1)
    return y

def generate_synthetic_data(
        num_train=1000,
        num_test=200,
        filename=DEFAULT_FILENAME,
        prepend_timestamp=False):
    # Create training data set
    x_train = np.random.randn(num_train, 2)
    y_train = classification_rule(x_train)

    # Create test data set
    x_test = np.random.randn(num_test, 2)
    y_test = classification_rule(x_test)

    # Add timestamp, if specified
    if prepend_timestamp:
        filename = time.strftime("%Y.%m.%d-%H.%M.%S-") + filename

    # Save npz file
    logging.info("Saving {}...".format(filename))
    np.savez(
        filename,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test)
    
    # TODO: return path to data

def load_data(filename=DEFAULT_FILENAME):
    with np.load(filename) as data:
        x_train = data["x_train"]
        y_train = data["y_train"]
        x_test = data["x_test"]
        y_test = data["y_test"]
    
    return x_train, y_train, x_test, y_test

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_synthetic_data()