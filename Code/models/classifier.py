import tensorflow as tf

class NeuralClassifier:
    def __init__(
        self,
        input_dim=2,
        num_hidden_units=3,
        hidden_layer_activation_function=tf.tanh,
        learning_rate=0.01,
        ):
        # Define network
        # TODO: add regulariser
        self.input_placeholder = tf.placeholder(
            dtype=tf.float32, shape=(None, input_dim))
        self.hidden_op = tf.layers.dense(
            inputs=self.input_placeholder,
            units=num_hidden_units,
            activation=hidden_layer_activation_function)
        self.logit_op = tf.layers.dense(inputs=self.hidden_op, units=1)
        self.predict_op = tf.sigmoid(self.logit_op)

        # Define loss and optimiser
        # TODO: add accuracy op
        self.labels_placeholder = tf.placeholder(
            dtype=tf.float32, shape=(None, 1))
        self.loss_op = tf.losses.sigmoid_cross_entropy(
            self.labels_placeholder, self.logit_op)
        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_op = self.optimizer.minimize(self.loss_op)
        # accuracy = tf.reduce_mean(tf.cast(tf.equal(y, y_true), tf.float32)) # need y>.5

        # Create op for initialising variables
        self.init_op = tf.global_variables_initializer()

        # Create summaries, for visualising in Tensorboard
        # TODO: separate train and test loss; visualise gradients
        # Training summaries:
        train_loss_summary = tf.summary.scalar("Train_loss", self.loss_op)
        train_hidden_layer_activations_summary = tf.summary.histogram(
            "Train_hidden_layer_activations", self.hidden_op
        )
        train_logits = tf.summary.histogram("Train_logits", self.logit_op)
        # tf.summary.scalar("Accuracy", accuracy)
        # tf.summary.histogram("Gradients", adam.compute_gradients(loss))
        self.train_summary_op = tf.summary.merge([
            train_loss_summary,
            train_hidden_layer_activations_summary,
            train_logits
        ])
        # Test summaries:
        test_loss_summary = tf.summary.scalar("Test_loss", self.loss_op)
        self.test_summary_op = tf.summary.merge([
            test_loss_summary,
            # <Other test summaries>
        ])
        

    def initialize_variables(self, sess):
        sess.run(self.init_op)

    def predict(self, sess, input_data):
        predictions = sess.run(
            self.predict_op,
            feed_dict={self.input_placeholder: input_data}
        )
        return predictions
    
    def training_step_with_progress(self, sess, x_train, y_train):
        train_loss_val, train_summary_val, _ = sess.run([
            self.loss_op,
            self.train_summary_op,
            self.train_op],
            feed_dict={
                self.input_placeholder: x_train,
                self.labels_placeholder: y_train
            }
        )
        return train_loss_val, train_summary_val

    def training_step_no_progress(self, sess, x_train, y_train):
        sess.run(
            self.train_op,
            feed_dict={
                self.input_placeholder: x_train,
                self.labels_placeholder: y_train
            }
        )
    
    def test_set_progress(self, sess, x_test, y_test):
        test_loss_val, test_summary_val = sess.run([
            self.loss_op,
            self.test_summary_op],
            feed_dict={
                self.input_placeholder: x_test,
                self.labels_placeholder: y_test
            }
        )
        return test_loss_val, test_summary_val