"""Tests for the Hungarian loss function."""

import pytest
import tensorflow as tf
from hungarian_loss.loss import (
    hungarian_loss,
    compute_euclidean_distance,
    HungarianLoss,
)

EPS = tf.constant(0.001, tf.float32)


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
    expected = tf.constant(3.254, tf.float32)
    assert actual.shape == expected.shape
    delta = tf.reduce_sum(tf.abs(actual - expected))
    assert tf.math.less(delta, EPS)


def test_hungarian_multipart_loss():
    """Tests the `hungarian_multipart_loss` function."""

    def rmse(y_true, y_pred):
        return tf.sqrt(tf.keras.losses.mse(y_pred, y_true))

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

    loss = HungarianLoss([4], 0, compute_euclidean_distance, [rmse], [1.0])
    actual = loss.call(y_true, y_pred)
    expected = tf.constant(3.254, tf.float32)
    assert actual.shape == expected.shape
    delta = tf.reduce_sum(tf.abs(actual - expected))
    assert tf.math.less(delta, EPS)

    loss = HungarianLoss(
        [4], 0, compute_euclidean_distance, [tf.keras.losses.mse], [1.0]
    )
    actual = loss.call(y_true, y_pred)
    expected = tf.constant(12.5, tf.float32)
    assert actual.shape == expected.shape
    delta = tf.reduce_sum(tf.abs(actual - expected))
    assert tf.math.less(delta, EPS)

    with pytest.raises(TypeError):
        HungarianLoss(
            None, 0, compute_euclidean_distance, [tf.keras.losses.mse], [1.0]
        )
    with pytest.raises(ValueError):
        HungarianLoss(
            [], 0, compute_euclidean_distance, [tf.keras.losses.mse], [1.0]
        )
    with pytest.raises(ValueError):
        HungarianLoss(
            [4], 1, compute_euclidean_distance, [tf.keras.losses.mse], [1.0]
        )

    HungarianLoss(
        [4], 0, compute_euclidean_distance, [tf.keras.losses.mse], None
    )
    with pytest.raises(ValueError):
        HungarianLoss([4], 0, compute_euclidean_distance, [], None)

    HungarianLoss([4], 0, compute_euclidean_distance, None, [1.0])
    with pytest.raises(ValueError):
        HungarianLoss([4], 0, compute_euclidean_distance, None, [])
