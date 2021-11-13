"""
Tests for the Hungarian algorithm steps.
"""

import tensorflow as tf

from hungarian_loss.steps import reduce_rows, reduce_cols, scratch_matrix


def test_reduce_rows():
    """Tests the `reduce_rows` function."""
    matrix = tf.constant(
        [[[30.0, 25.0, 10.0], [15.0, 10.0, 20.0], [25.0, 20.0, 15.0]]]
    )
    actual = reduce_rows(matrix)
    expected = tf.constant(
        [[[20.0, 15.0, 0.0], [5.0, 0.0, 10.0], [10.0, 5.0, 0.0]]], tf.float16
    )
    assert tf.reduce_all(tf.equal(actual, expected))


def test_reduce_cols():
    """Tests the `reduce_cols` function."""
    matrix = tf.constant(
        [[[30.0, 25.0, 10.0], [15.0, 10.0, 20.0], [25.0, 20.0, 15.0]]]
    )
    actual = reduce_cols(matrix)
    expected = tf.constant(
        [[[15.0, 15.0, 0.0], [0.0, 0.0, 10.0], [10.0, 10.0, 5.0]]], tf.float16
    )
    assert tf.reduce_all(tf.equal(actual, expected))


def test_scratch_matrix():
    """Tests the `scratch_matrix` function."""
    matrix = tf.Variable(
        [[[15.0, 15.0, 0.0], [0.0, 0.0, 10.0], [5.0, 5.0, 0.0]]],
        dtype=tf.float16,
    )
    actual_row_mask, actual_col_mask = scratch_matrix(matrix)
    expected_row_mask = tf.constant([[False], [True], [False]], tf.bool)
    expected_col_mask = tf.constant([[False, False, True]])
    assert tf.reduce_all(tf.equal(actual_row_mask, expected_row_mask))
    assert tf.reduce_all(tf.equal(actual_col_mask, expected_col_mask))
