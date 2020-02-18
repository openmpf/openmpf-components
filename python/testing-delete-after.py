import numpy as np
import cv2
import tensorflow as tf

from kaffe.tensorflow import Network


data_path = "tfrecord_all_blend"
num_epochs = 100000
batch_size = 4 # 8
max_steps = 500000
model_dir = "./training/sanity_all_blend_model"
save_rate = 25000
epoch_size = 5000
#learning_rate=0.00001

run_config = tf.estimator.RunConfig()
run_config = run_config.replace(keep_checkpoint_max   = None)
run_config = run_config.replace(save_checkpoints_steps = save_rate)
run_config = run_config.replace(save_checkpoints_secs  = None)

class TrainingConfig(object):
  """Wrapper class for training hyperparameters."""

  def __init__(self):
    """Sets the default training hyperparameters."""
    self.num_examples_per_epoch = epoch_size
    # Optimizer for training the model.
    self.optimizer = "SGD"

    # Learning rate for the initial phase of training.
    self.initial_learning_rate = 0.001
    self.learning_rate_decay_factor = 0.1
    self.num_epochs_per_decay = 20.0

    # Learning rate when fine tuning the Inception v3 parameters.
    #self.train_inception_learning_rate = 0.0005

    # If not None, clip gradients to this value.
    self.clip_gradients = None#5.0


class CaffeNet(Network):
    def setup(self):
        (self.feed('data')
             .conv(11, 11, 96, 4, 4, padding='VALID', name='conv1')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
             .lrn(2, 1.99999994948e-05, 0.75, name='norm1')
             .conv(5, 5, 256, 1, 1, group=2, name='conv2')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
             .lrn(2, 1.99999994948e-05, 0.75, name='norm2')
             .conv(3, 3, 384, 1, 1, name='conv3')
             .conv(3, 3, 384, 1, 1, group=2, name='conv4')
             .conv(3, 3, 256, 1, 1, group=2, name='conv5')
             .max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
             .fc(4096, name='fc6')
             .fc(4096, name='fc7')
             .fc(1, relu=False, name='fine_fc8')
             .sigmoid(name='prob_att'))
             #.softmax_cross_entropy_loss(labels=labels, name='softmax_loss'))

def deepMAR(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    # Reshape X to 4-D tensor: [batch_size, width, height, channels]
    # May not be needed
    # Images are 28x28 pixels, and have three color channels.
    input_layer = features#features["x"]#tf.reshape(features["x"], [-1, 227, 227, 3])



    net = CaffeNet({'data': input_layer})
    logits = net.layers['prob_att']
    # Calculate Loss (for both TRAIN and EVAL modes)
    weights = np.array([0.5], dtype='float32')
   # weights = np.exp(-weights)
    weights = tf.constant(weights)
    weights = tf.reshape(weights, (1,1))
    loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=logits, weights=weights)
    #cross_entropy = tf.reduce_mean(loss)
    predictions = {
          # Generate predictions (for PREDICT and EVAL mode)
          "probabilities": logits
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        training_config = TrainingConfig()
        #optimizer = tf.train.AdamOptimizer(learning_rate=training_config.initial_learning_rate)


        #BROKEN train_op = optimizer.minimize(cross_entropy#)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=training_config.initial_learning_rate)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())

        num_batches_per_epoch = (training_config.num_examples_per_epoch/batch_size)
        decay_steps = int(num_batches_per_epoch *
                          training_config.num_epochs_per_decay)


       # def _learning_rate_decay_fn(learning_rate, global_step):
       #     return tf.train.exponential_decay(
       #           learning_rate,
       #           global_step,
       #           decay_steps=decay_steps,
       #           decay_rate=training_config.learning_rate_decay_factor,
       #           staircase=True)

        #learning_rate_decay_fn = _learning_rate_decay_fn


        #train_op = tf.contrib.layers.optimize_loss(
        #    loss=loss,
        #    global_step=tf.train.get_global_step(),
        #    learning_rate=training_config.initial_learning_rate,
        #    optimizer=training_config.optimizer,
        #    clip_gradients=training_config.clip_gradients,
        #    learning_rate_decay_fn=learning_rate_decay_fn)

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

      # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


