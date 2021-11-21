"""Tests for the Hungarian loss function."""

import tensorflow as tf
from hungarian_loss.loss import (
    hungarian_loss,
    compute_hungarian_loss_on_distance_matrix,
)

EPS = tf.constant(0.001, tf.float16)


def test_compute_hungarian_loss_on_distance_matrix():
    """Tests the `compute_hungarian_loss_on_distance_matrix` function."""
    dist = tf.constant([[3.742, 2.45], [11.23, 9.27]], tf.float16)
    actual = compute_hungarian_loss_on_distance_matrix(dist)
    expected = tf.constant(3.254, tf.float16)
    delta = tf.reduce_sum(tf.abs(actual - expected))
    assert tf.math.less(delta, EPS)


def test_hungarian_loss():
    """Tests the `hungarian_loss` function."""
    y_true = tf.constant(
        [
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        ]
    )
    y_pred = tf.constant(
        [
            [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]],
            [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]],
        ]
    )
    actual = hungarian_loss(y_true, y_pred)
    expected = tf.constant([3.254, 3.254], tf.float16)
    delta = tf.reduce_sum(tf.abs(actual - expected))
    assert tf.math.less(delta, EPS)
