print("Starting imports...")
import tensorflow as tf
print("Imported tf")
import time
import logging
from data.data_util import load_data, DEFAULT_FILENAME
from models.classifier import NeuralClassifier
print("Finished imports")

DEFAULT_SAVE_DIR = \
    "models/" + time.strftime("%Y.%m.%d-%H.%M.%S-") + "saved_model"

def train(
    num_epochs=2000,
    print_every=100,
    data_filename=DEFAULT_FILENAME,
    savedir=DEFAULT_SAVE_DIR
):
    logging.info("Loading data...")
    x_train, y_train, x_test, y_test = load_data(data_filename)
    logging.info("Creating model...")
    model = NeuralClassifier()


    with tf.Session() as sess:
        model.initialize_variables(sess)
        for e in range(num_epochs):
            loss_val, summary_val, _ = sess.run(
                (model.loss_op,
                model.merged_summary_op,
                model.train_op),
                feed_dict={
                    model.input_placeholder: x_train,
                    model.output_placeholder: y_train})
            print("Epoch: {:<8} | Loss: {:<.6f}".format(e, loss_val))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train(num_epochs=10,print_every=1)
