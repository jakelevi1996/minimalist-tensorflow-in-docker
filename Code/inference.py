import tensorflow as tf
import numpy as np
import logging
# Before importing pyplot, set the matplotlib backend to allow usage in Docker container
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def infer(model, saved_model_dir, x_test):
    saver = tf.train.Saver()

    with tf.Session() as sess:
        logging.info("Restoring model...")
        saver.restore(sess, saved_model_dir)
        logging.info("Making predictions...")
        y_pred = model.predict(sess, x_test)
    
    return y_pred


def plot_predictions(
    model, saved_model_dir,
    x_train, y_train,
    x_test, y_test,
    saved_image_path="results/classification-results.png"
):
    logging.info("Plotting results...")
    plt.plot(
        x_train[y_train[:,0]==0, 0], x_train[y_train[:,0]==0, 1], 'bo',
        x_train[y_train[:,0]==1, 0], x_train[y_train[:,0]==1, 1], 'ro',
        x_test[y_test[:,0]==0, 0], x_test[y_test[:,0]==0, 1], 'bx',
        x_test[y_test[:,0]==1, 0], x_test[y_test[:,0]==1, 1], 'rx',
        alpha=.1
    )
    
    # Create grid for evaluation of test set
    x_array = np.linspace(-4, 4, 100)
    xx0, xx1 = np.meshgrid(x_array, x_array)
    
    x_grid = np.concatenate(
        (xx0.reshape(-1,1), xx1.reshape(-1,1)),
        axis=1
    )
    
    y_grid = infer(model, saved_model_dir, x_grid)

    plt.contour(
        xx0, xx1, y_grid.reshape(xx0.shape),
        [.2, .4, .6, .8], cmap='bwr')
    plt.grid(True)
    plt.axis('equal')

    logging.info("Saving figure...")
    plt.savefig(saved_image_path)
