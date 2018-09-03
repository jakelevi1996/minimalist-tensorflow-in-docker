import tensorflow as tf
import logging
import time
from data.data_util import load_data, DEFAULT_FILENAME
from models.classifier import NeuralClassifier



def display_progress(epoch, train_loss_val, test_loss_val=None):
    logging.info("{}: {:<8} | {}: {:<.6f} | {}: {:<.6f}".format(
        "Epoch", epoch,
        "Train loss", train_loss_val,
        "Test loss", test_loss_val,
    ))

def get_save_dir():
    pass

def training_condition(
    epoch, num_epochs,
    start_time, num_seconds):
    pass

def train(
    model,
    x_train, y_train,
    x_test, y_test,
    num_epochs=3000,
    num_seconds=None,
    print_every=1000,
    model_name="standard_model",
    saved_model_dir=None,
    log_dir=None
    ):
    # Create names for saving of models and summaries, if not specified
    timestamp = time.strftime("%Y.%m.%d-%H.%M.%S-")
    if saved_model_dir is None:
        saved_model_dir = "models/" + timestamp + model_name + "/"
    logging.info("Saving models in " + saved_model_dir)
    if log_dir is None:
        log_dir = "results/" + timestamp + model_name + "/"
    logging.info("Saving tensorboard summaries in " + log_dir)

    # TODO: add option to load model for continued training
    logging.info("Creating Saver object...")
    saver = tf.train.Saver()

    logging.info("Creating session...")
    with tf.Session() as sess:
        logging.info("Initialising variables...")
        model.initialize_variables(sess)
        logging.info("Creating FileWriter for Tensorboard...")
        writer = tf.summary.FileWriter(log_dir, sess.graph)

        logging.info("Entering training loop...")
        for epoch in range(num_epochs):
            # Evaluate graph, summaries, and training op
            train_loss_val, train_summary_val = \
                model.training_step_with_progress(sess, x_train, y_train)
            test_loss_val, test_summary_val = \
                model.test_set_progress(sess, x_test, y_test)
            # Add summaries for Tensorboard
            writer.add_summary(train_summary_val, epoch)
            writer.add_summary(test_summary_val, epoch)
            # Display progress at specified intervals
            if epoch % print_every == 0:
                display_progress(epoch, train_loss_val, test_loss_val)
        
        # End of training loop
        logging.info("Evaluating final loss...")
        train_loss_val, summary_val = model.training_step_with_progress(
            sess, x_train, y_train)
        writer.add_summary(summary_val, num_epochs)
        display_progress(num_epochs, train_loss_val, test_loss_val)

        logging.info("Saving model...")
        # `save` method might save results in a different directory,
        # depending on the format of the string
        saved_model_dir = saver.save(sess, saved_model_dir)
    
    return saved_model_dir



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    logging.info("Loading data...")
    x_train, y_train, x_test, y_test = load_data(DEFAULT_FILENAME)
    logging.info("Creating model...")
    model = NeuralClassifier()

    train(model, x_train, y_train, x_test, y_test)
