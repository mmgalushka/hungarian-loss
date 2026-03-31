"""The helpful constants."""

import tensorflow as tf

ZERO = tf.constant(0, tf.float32)
"""The constant for 0."""
ONE = tf.constant(1, tf.float32)
"""The constant for 1."""
EPSILON = tf.constant(1e-4, tf.float32)
"""Small value added inside sqrt to keep gradients bounded at zero distance."""
