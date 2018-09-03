import tensorflow as tf
import logging
import time
from data.data_util import load_data, DEFAULT_FILENAME
from models.classifier import NeuralClassifier



def display_progress(epoch, loss_val):
    logging.info("Epoch: {:<8} | Train loss: {:<.6f} | Test loss: ".format(
        epoch, loss_val))


def train(
    num_epochs=5000,
    print_every=1000,
    model_name="standard_model",
    data_filename=DEFAULT_FILENAME,
    save_dir=None,
    log_dir=None
    ):
    # Create names for saving of models and summaries, if not specified
    timestamp = time.strftime("%Y.%m.%d-%H.%M.%S-")
    if save_dir is None:
        save_dir = "models/" + timestamp + model_name + "/saved_model"
    logging.info("Saving models in " + save_dir)
    if log_dir is None:
        log_dir = "models/" + timestamp + model_name + "/logs"
    logging.info("Saving tensorboard summaries in " + log_dir)

    # TODO: add option to load model for continued training
    logging.info("Loading data...")
    x_train, y_train, x_test, y_test = load_data(data_filename)
    logging.info("Creating model...")
    model = NeuralClassifier()

    logging.info("Creating session...")
    with tf.Session() as sess:
        logging.info("Initialising variables...")
        model.initialize_variables(sess)
        logging.info("Creating FileWriter for Tensorboard...")
        writer = tf.summary.FileWriter(log_dir, sess.graph)

        logging.info("Entering training loop...")
        for epoch in range(num_epochs):
            # Evaluate graph, summaries, and training op
            train_loss_val, summary_val, _ = sess.run([
                model.loss_op,
                model.merged_summary_op,
                model.train_op],
                feed_dict={
                    model.input_placeholder: x_train,
                    model.output_placeholder: y_train})
            writer.add_summary(summary_val, epoch)
            if epoch % print_every == 0:
                display_progress(epoch, train_loss_val)
        logging.info("Evaluating final loss...")
        train_loss_val, summary_val = sess.run([
            model.loss_op,
            model.merged_summary_op],
            feed_dict={
                model.input_placeholder: x_train,
                model.output_placeholder: y_train})
        writer.add_summary(summary_val, num_epochs)
        display_progress(num_epochs, train_loss_val)



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # train(num_epochs=10,print_every=1)
    train()
