"""
The tests for the Hungarian loss function.
"""

import tensorflow as tf
from hungarian_loss.main import (
    euclidean_distance,
    pairs_mesh,
    hungarian_mask,
    hungarian_loss,
)

EPS = tf.constant(0.001)


def test_euclidean_distance():
    """Tests the euclidean_distance function."""
    a = tf.constant(
        [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]
    )  # pylint: disable=invalid-name

    b = tf.constant(
        [[[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]]]
    )  # pylint: disable=invalid-name

    actual = euclidean_distance(a, b)
    expected = tf.constant([[[3.7416575, 2.4494898], [11.224972, 9.273619]]])

    delta = tf.reduce_sum(tf.abs(actual - expected))
    assert tf.math.less(delta, EPS)


def test_pairs_mesh():
    """Tests the pairs_mesh function."""
    n = 2  # pylint: disable=invalid-name

    actual = pairs_mesh(n)
    expected = tf.constant([[0, 0], [0, 1], [1, 0], [1, 1]])

    delta = tf.reduce_sum(tf.abs(actual - expected))
    assert tf.math.less(tf.cast(delta, tf.float32), EPS)


def test_hungarian_mask():
    """Tests the hungarian_mask function."""
    cost = tf.constant([[[3.7416575, 2.4494898], [11.224972, 9.273619]]])

    actual = hungarian_mask(cost)
    expected = tf.constant([[0, 1], [1, 0]])

    delta = tf.reduce_sum(tf.abs(actual - expected))
    assert tf.math.less(tf.cast(delta, tf.float32), EPS)


def test_hungarian_loss():
    """Tests the euclidean_distance function."""
    a = tf.constant(
        [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]]
    )  # pylint: disable=invalid-name

    b = tf.constant(
        [[[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]]]
    )  # pylint: disable=invalid-name

    actual = hungarian_loss(a, b)
    expected = tf.constant([13.674461])

    delta = tf.reduce_sum(tf.abs(actual - expected))
    assert tf.math.less(delta, EPS)
