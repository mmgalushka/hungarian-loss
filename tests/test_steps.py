"""
Tests for the Hungarian algorithm steps.
"""

import tensorflow as tf

from hungarian_loss.steps import (
    reduce_rows,
    reduce_cols,
    scratch_matrix,
    is_optimal_assignment,
    shift_zeros,
    reduce_matrix,
)


def test_reduce_rows():
    """Tests the `reduce_rows` function."""
    matrix = tf.constant(
        [[[30.0, 25.0, 10.0], [15.0, 10.0, 20.0], [25.0, 20.0, 15.0]]],
        tf.float16,
    )
    actual = reduce_rows(matrix)
    expected = tf.constant(
        [[[20.0, 15.0, 0.0], [5.0, 0.0, 10.0], [10.0, 5.0, 0.0]]], tf.float16
    )
    assert tf.reduce_all(tf.equal(actual, expected))


def test_reduce_cols():
    """Tests the `reduce_cols` function."""
    matrix = tf.constant(
        [[[30.0, 25.0, 10.0], [15.0, 10.0, 20.0], [25.0, 20.0, 15.0]]],
        tf.float16,
    )
    actual = reduce_cols(matrix)
    expected = tf.constant(
        [[[15.0, 15.0, 0.0], [0.0, 0.0, 10.0], [10.0, 10.0, 5.0]]], tf.float16
    )
    assert tf.reduce_all(tf.equal(actual, expected))


def test_scratch_matrix():
    """Tests the `scratch_matrix` function."""
    matrix = tf.constant(
        [[[15.0, 15.0, 0.0], [0.0, 0.0, 10.0], [5.0, 5.0, 0.0]]], tf.float16
    )
    actual_row_mask, actual_col_mask = scratch_matrix(matrix)
    expected_row_mask = tf.constant([[False], [True], [False]], tf.bool)
    expected_col_mask = tf.constant([[False, False, True]], tf.bool)
    assert tf.reduce_all(tf.equal(actual_row_mask, expected_row_mask))
    assert tf.reduce_all(tf.equal(actual_col_mask, expected_col_mask))


def test_is_optimal_assignment():
    """Tests the `is_optimal_assignment` function."""
    rows_mask = tf.constant([[False], [True], [False]], tf.bool)
    cols_mask = tf.constant([[True, False, True]], tf.bool)
    actual = is_optimal_assignment(rows_mask, cols_mask)
    expected = tf.constant(True, tf.bool)
    assert tf.equal(actual, expected)

    rows_mask = tf.constant([[False], [True], [False]], tf.bool)
    cols_mask = tf.constant([[False, False, True]], tf.bool)
    actual = is_optimal_assignment(rows_mask, cols_mask)
    expected = tf.constant(False, tf.bool)
    assert tf.equal(actual, expected)


def test_shift_zeros():
    """Tests the `shift_zeros` function."""
    matrix = tf.constant(
        [[[15.0, 15.0, 0.0], [0.0, 0.0, 10.0], [5.0, 5.0, 0.0]]], tf.float16
    )
    scratched_rows_mask = tf.constant([[False], [True], [False]], tf.bool)
    scratched_cols_mask = tf.constant([[False, False, True]], tf.bool)
    (
        actual_matrix,
        actual_scratched_rows_mask,
        actual_scratched_cols_mask,
    ) = shift_zeros(matrix, scratched_rows_mask, scratched_cols_mask)
    expected_matrix = tf.constant(
        [[[10.0, 10.0, 0.0], [0.0, 0.0, 15.0], [0.0, 0.0, 0.0]]], tf.float16
    )
    expected_scratched_rows_mask = tf.constant(
        [[False], [True], [False]], tf.bool
    )
    expected_scratched_cols_mask = tf.constant([[False, False, True]], tf.bool)
    assert tf.reduce_all(tf.equal(actual_matrix, expected_matrix))
    assert tf.reduce_all(
        tf.equal(actual_scratched_rows_mask, expected_scratched_rows_mask)
    )
    assert tf.reduce_all(
        tf.equal(actual_scratched_cols_mask, expected_scratched_cols_mask)
    )


def test_reduce_matrix():
    """Tests the `reduce_matrix` function."""
    matrix = tf.constant(
        [[[30.0, 25.0, 10.0], [15.0, 10.0, 20.0], [25.0, 20.0, 15.0]]],
        tf.float16,
    )
    actual_matrix = reduce_matrix(matrix)
    expected_matrix = tf.constant(
        [[[10.0, 10.0, 0.0], [0.0, 0.0, 15.0], [0.0, 0.0, 0.0]]], tf.float16
    )
    assert tf.reduce_all(tf.equal(actual_matrix, expected_matrix))
