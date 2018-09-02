import numpy as np
import matplotlib.pyplot as plt

DEFAULT_FILENAME = "synthetic_data.npz"

def classification_rule(x):
    bool_val = (x[:,0]**2 + x[:,1]**2) < 1
    y = (1 * bool_val).reshape(-1,1)
    return y

def generate_synthetic_data(
        num_train=1000,
        filename=DEFAULT_FILENAME):
    # Create training data set
    x_train = np.random.randn(num_train, 2)
    y_train = classification_rule(x_train)

    # Create grid for evaluation of test set
    x_array = np.linspace(-4, 4, 100)
    xx0, xx1 = np.meshgrid(x_array, x_array)
    
    # Create test data set
    x_test = np.concatenate(
        (xx0.reshape(-1,1), xx1.reshape(-1,1)),
        axis=1)
    y_test = classification_rule(x_test)

    # Save npz file
    np.savez(
        filename,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test)

def load_data(filename=DEFAULT_FILENAME):
    with np.load(filename) as data:
        x_train = data["x_train"]
        y_train = data["y_train"]
        x_test = data["x_test"]
        y_test = data["y_test"]
    
    return x_train, y_train, x_test, y_test
        

x = np.arange(6).reshape(-1,2)
y = classification_rule(x)
print(x)
print(y)
print(type(x))
print(type(y))
generate_synthetic_data()
with np.load("synthetic_data.npz") as data:
    for i in data:
        print(i)
    