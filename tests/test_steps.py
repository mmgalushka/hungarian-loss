"""Tests for the Hungarian algorithm steps."""

import tensorflow as tf

from hungarian_loss.steps import (
    compute_euclidean_distance,
    reduce_rows,
    reduce_cols,
    scratch_matrix,
    is_optimal_assignment,
    shift_zeros,
    reduce_matrix,
    select_optimal_assignment_mask,
)

EPS = tf.constant(0.001, tf.float32)


def test_compute_euclidean_distance():
    """Tests the `compute_euclidean_distance` function."""
    a = tf.constant(
        [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    )  # pylint: disable=invalid-name

    b = tf.constant(
        [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]]
    )  # pylint: disable=invalid-name

    actual = compute_euclidean_distance(a, b)
    expected = tf.constant(
        [[3.7416575, 2.4494898], [11.224972, 9.273619]], tf.float32
    )

    delta = tf.reduce_sum(tf.abs(actual - expected))
    assert tf.less(delta, EPS)


def test_reduce_rows():
    """Tests the `reduce_rows` function."""
    matrix = tf.constant(
        [[30.0, 25.0, 10.0], [15.0, 10.0, 20.0], [25.0, 20.0, 15.0]],
        tf.float32,
    )
    actual = reduce_rows(matrix)
    expected = tf.constant(
        [[20.0, 15.0, 0.0], [5.0, 0.0, 10.0], [10.0, 5.0, 0.0]], tf.float32
    )
    assert tf.reduce_all(tf.equal(actual, expected))


def test_reduce_cols():
    """Tests the `reduce_cols` function."""
    matrix = tf.constant(
        [[30.0, 25.0, 10.0], [15.0, 10.0, 20.0], [25.0, 20.0, 15.0]],
        tf.float32,
    )
    actual = reduce_cols(matrix)
    expected = tf.constant(
        [[15.0, 15.0, 0.0], [0.0, 0.0, 10.0], [10.0, 10.0, 5.0]], tf.float32
    )
    assert tf.reduce_all(tf.equal(actual, expected))


def test_scratch_matrix():
    """Tests the `scratch_matrix` function."""
    matrix = tf.constant(
        [[15.0, 15.0, 0.0], [0.0, 0.0, 10.0], [5.0, 5.0, 0.0]], tf.float32
    )
    actual_row_mask, actual_col_mask = scratch_matrix(matrix)
    expected_row_mask = tf.constant([[False], [True], [False]], tf.bool)
    assert tf.reduce_all(tf.equal(actual_row_mask, expected_row_mask))
    expected_col_mask = tf.constant([[False, False, True]], tf.bool)
    assert tf.reduce_all(tf.equal(actual_col_mask, expected_col_mask))
    matrix = tf.constant(
        [[15.0, 15.0, 0.0], [0.0, 0.0, 10.0], [5.0, 5.0, 0.0]], tf.float32
    )


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

    rows_mask = tf.constant([[True], [True], [True]], tf.bool)
    cols_mask = tf.constant([[True, True, True]], tf.bool)
    actual = is_optimal_assignment(rows_mask, cols_mask)
    expected = tf.constant(True, tf.bool)
    assert tf.equal(actual, expected)

    rows_mask = tf.constant([[False], [False], [False]], tf.bool)
    cols_mask = tf.constant([[False, False, False]], tf.bool)
    actual = is_optimal_assignment(rows_mask, cols_mask)
    expected = tf.constant(False, tf.bool)
    assert tf.equal(actual, expected)


def test_shift_zeros():
    """Tests the `shift_zeros` function."""
    matrix = tf.constant(
        [[15.0, 15.0, 0.0], [0.0, 0.0, 10.0], [5.0, 5.0, 0.0]], tf.float32
    )
    scratched_rows_mask = tf.constant([[False], [True], [False]], tf.bool)
    scratched_cols_mask = tf.constant([[False, False, True]], tf.bool)
    (
        actual_matrix,
        actual_scratched_rows_mask,
        actual_scratched_cols_mask,
    ) = shift_zeros(matrix, scratched_rows_mask, scratched_cols_mask)
    expected_matrix = tf.constant(
        [[10.0, 10.0, 0.0], [0.0, 0.0, 15.0], [0.0, 0.0, 0.0]], tf.float32
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
        [[30.0, 25.0, 10.0], [15.0, 10.0, 20.0], [25.0, 20.0, 15.0]],
        tf.float32,
    )
    actual_matrix = reduce_matrix(matrix)
    expected_matrix = tf.constant(
        [[10.0, 10.0, 0.0], [0.0, 0.0, 15.0], [0.0, 0.0, 0.0]], tf.float32
    )
    assert tf.reduce_all(tf.equal(actual_matrix, expected_matrix))


def test_select_optimal_assignment_mask():
    """Tests the `select_optimal_assignment_mask` function."""
    matrix = tf.constant(
        [[10.0, 10.0, 0.0], [0.0, 0.0, 15.0], [0.0, 0.0, 0.0]], tf.float32
    )
    actual_mask = select_optimal_assignment_mask(matrix)
    expected_mask_1 = tf.constant(
        [[False, False, True], [True, False, False], [False, True, False]],
        tf.bool,
    )
    expected_mask_2 = tf.constant(
        [[False, False, True], [False, True, False], [True, False, False]],
        tf.bool,
    )
    assert tf.logical_or(
        tf.reduce_all(tf.equal(actual_mask, expected_mask_1)),
        tf.reduce_all(tf.equal(actual_mask, expected_mask_2)),
    )
