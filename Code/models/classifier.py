import tensorflow as tf

def parse_name(name):
    """
    Used to make sure that the name passed to `tf.summary.histogram` does not
    contain any illegal characters.
    Args:
    - name: name to be parsed
    Returns:
    - name: parsed name with no illegal characters
    """
    while ":" in name:
        name = name[:-1]
    return name

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
            dtype=tf.float32,
            shape=(None, input_dim),
            name="Input"
        )
        self.hidden_op = tf.layers.dense(
            inputs=self.input_placeholder,
            units=num_hidden_units,
            activation=hidden_layer_activation_function,
            name="Input_layer"
        )
        self.logit_op = tf.layers.dense(
            inputs=self.hidden_op,
            units=1,
            name="Hidden_layer"
        )
        self.predict_op = tf.sigmoid(
            self.logit_op,
            name="Prediction"
        )

        # Define loss and optimiser
        # TODO: add accuracy op
        self.labels_placeholder = tf.placeholder(
            dtype=tf.float32, shape=(None, 1), name="Labels"
        )
        self.loss_op = tf.losses.sigmoid_cross_entropy(
            self.labels_placeholder, self.logit_op
        )
        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_op = self.optimizer.minimize(self.loss_op)
        # accuracy = tf.reduce_mean(tf.cast(tf.equal(y, y_true), tf.float32)) # need y>.5

        # Create op for initialising variables
        self.init_op = tf.global_variables_initializer()

        # Create summaries, for visualising in Tensorboard
        # Training summaries:
        # tf.summary.scalar("Accuracy", accuracy)
        variables = tf.trainable_variables()
        gradients = tf.gradients(self.loss_op, variables)
        self.train_summary_op = tf.summary.merge([
            tf.summary.scalar("Train_loss", self.loss_op),
            *[tf.summary.histogram(parse_name(v.name), v) for v in variables],
            *[tf.summary.histogram(parse_name(g.name), g) for g in gradients]
        ])
        # Test summaries:
        self.test_summary_op = tf.summary.merge([
            tf.summary.scalar("Test_loss", self.loss_op),
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
    
    def training_step_with_summary(self, sess, x_train, y_train):
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

    def training_step_no_summary(self, sess, x_train, y_train):
        sess.run(
            self.train_op,
            feed_dict={
                self.input_placeholder: x_train,
                self.labels_placeholder: y_train
            }
        )
    
    def test_set_summary(self, sess, x_test, y_test):
        test_loss_val, test_summary_val = sess.run([
            self.loss_op,
            self.test_summary_op],
            feed_dict={
                self.input_placeholder: x_test,
                self.labels_placeholder: y_test
            }
        )
        return test_loss_val, test_summary_val