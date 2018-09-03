import tensorflow as tf
import logging
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
    x_train, y_train,
    x_test, y_pred,
    saved_image_path="results/classification-results.png"
):
    logging.info("Plotting results...")
    plt.plot(
        x_train[y_train[:,0]==0, 0], x_train[y_train[:,0]==0, 1], 'bo',
        x_train[y_train[:,0]==1, 0], x_train[y_train[:,0]==1, 1], 'ro',
        alpha=.1
    )

    x0 = x_test[:,0].reshape(100, 100)
    x1 = x_test[:,1].reshape(100, 100)
    y_pred = y_pred.reshape(100, 100)
    print(x0.shape, y_pred.shape)
    
    plt.contour(
        x0, x1, y_pred,
        [.2, .4, .6, .8], cmap='bwr')
    plt.grid(True)
    plt.axis('equal')

    logging.info("Saving figure...")
    plt.savefig(saved_image_path)
