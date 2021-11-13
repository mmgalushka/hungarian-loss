"""
The package implementing the loss function using the Hungarian algorithm.
"""

import tensorflow as tf

# from .main import ga_loss

ZERO = tf.constant(0, tf.float16)
ONE = tf.constant(1, tf.float16)
