import logging
logging.basicConfig(level=logging.INFO)
logging.info("Importing tensorflow...")
import tensorflow as tf
import time
from data.data_util import load_data, DEFAULT_FILENAME
from models.classifier import NeuralClassifier

timestamp = time.strftime("%Y.%m.%d-%H.%M.%S-")
DEFAULT_SAVE_DIR = "models/" + timestamp + "saved_model"
DEFAULT_LOG_DIR = "logs"

def display_progress(epoch, loss_val):
    logging.info("Epoch: {:<8} | Loss: {:<.6f}".format(epoch, loss_val))


def train(
    num_epochs=5000,
    print_every=1000,
    model_name="standard_model",
    data_filename=DEFAULT_FILENAME,
    savedir=DEFAULT_SAVE_DIR,
    logdir=DEFAULT_LOG_DIR
    ):
    # TODO: add option to load model for continued training
    logging.info("Loading data...")
    x_train, y_train, x_test, y_test = load_data(data_filename)
    logging.info("Creating model...")
    model = NeuralClassifier()


    with tf.Session() as sess:
        model.initialize_variables(sess)
        for epoch in range(num_epochs):
            # Evaluate graph, summaries, and training op
            loss_val, summary_val, _ = sess.run(
                (model.loss_op,
                model.merged_summary_op,
                model.train_op),
                feed_dict={
                    model.input_placeholder: x_train,
                    model.output_placeholder: y_train})
            if epoch % print_every == 0:
                display_progress(epoch, loss_val)



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # train(num_epochs=10,print_every=1)
    train()
