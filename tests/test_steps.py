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
        [
            [0.0, 5.0, 5.0, 5.0],
            [5.0, 0.0, 10.0, 5.0],
            [10.0, 10.0, 0.0, 15.0],
            [10.0, 5.0, 5.0, 0.0],
        ],
        tf.float32,
    )
    actual_row_mask, actual_col_mask = scratch_matrix(matrix)
    expected_row_mask = tf.constant(
        [[True], [False], [True], [False]], tf.bool
    )
    assert tf.reduce_all(tf.equal(actual_row_mask, expected_row_mask))
    expected_col_mask = tf.constant([[False, True, False, True]], tf.bool)
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
    # Test scenario 1
    matrix = tf.constant(
        [[10.0, 10.0, 0.0], [0.0, 0.0, 15.0], [0.0, 0.0, 0.0]], tf.float32
    )
    actual_mask = select_optimal_assignment_mask(matrix)
    expected_mask = tf.constant(
        [[False, False, True], [True, False, False], [False, True, False]],
        tf.bool,
    )
    assert tf.reduce_all(tf.equal(actual_mask, expected_mask))

    # Test scenario 2
    matrix = tf.constant(
        [
            [0.0, 1.0, 1.0, 0.0],
            [10.0, 0.0, 0.0, 10.0],
            [10.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
        ],
        tf.float32,
    )
    actual_mask = select_optimal_assignment_mask(matrix)
    expected_mask = tf.constant(
        [
            [True, False, False, False],
            [False, False, True, False],
            [False, True, False, False],
            [False, False, False, True],
        ],
        tf.bool,
    )
    assert tf.reduce_all(tf.equal(actual_mask, expected_mask))


def test_reduce_matrix_all_nan_terminates():
    """Bug 2: reduce_matrix returns finite result on all-NaN 2x2 input."""
    nan = float("nan")
    matrix = tf.constant([[nan, nan], [nan, nan]], tf.float32)
    result = reduce_matrix(matrix)
    assert tf.reduce_all(tf.math.is_finite(result))


def test_reduce_matrix_partial_nan_terminates():
    """Bug 2: reduce_matrix returns finite result on partial-NaN 3x3 input."""
    nan = float("nan")
    matrix = tf.constant(
        [[nan, 1.0, 2.0], [3.0, nan, 1.0], [2.0, 1.0, nan]], tf.float32
    )
    result = reduce_matrix(matrix)
    assert tf.reduce_all(tf.math.is_finite(result))


def test_reduce_matrix_partial_nan_valid_assignment():
    """Bug 2: end-to-end assignment is valid (one True per row and col) on
    partial-NaN input."""
    nan = float("nan")
    matrix = tf.constant(
        [[nan, 1.0, 2.0], [3.0, nan, 1.0], [2.0, 1.0, nan]], tf.float32
    )
    mask = select_optimal_assignment_mask(reduce_matrix(matrix))
    row_counts = tf.reduce_sum(tf.cast(mask, tf.int32), axis=1)
    col_counts = tf.reduce_sum(tf.cast(mask, tf.int32), axis=0)
    assert tf.reduce_all(tf.equal(row_counts, 1))
    assert tf.reduce_all(tf.equal(col_counts, 1))


def test_compute_euclidean_distance_identical_vectors_no_nan():
    """Bug 1a: no NaN when a == b (float32 cancellation repro, seed=0,
    dim=16)."""
    tf.random.set_seed(0)
    a = tf.random.uniform((4, 16), dtype=tf.float32)
    result = compute_euclidean_distance(a, a)
    assert not tf.reduce_any(tf.math.is_nan(result))
    diagonal = tf.linalg.diag_part(result)
    # EPSILON=1e-4 gives sqrt(0 + 1e-4) = 0.01 for identical vectors.
    assert tf.reduce_all(tf.less(diagonal, tf.constant(0.02, tf.float32)))


def test_compute_euclidean_distance_near_identical_no_nan():
    """Bug 1a: no NaN when a ≈ b (near-cancellation guard)."""
    tf.random.set_seed(0)
    a = tf.random.uniform((4, 16), dtype=tf.float32)
    b = a + tf.constant(1e-7, tf.float32)
    result = compute_euclidean_distance(a, b)
    assert not tf.reduce_any(tf.math.is_nan(result))


def test_compute_euclidean_distance_finite_gradient_at_zero():
    """Bug 1b: gradient is finite when distance is exactly zero."""
    a_var = tf.Variable([[1.0, 2.0, 3.0]])
    b_const = tf.constant([[1.0, 2.0, 3.0]])
    with tf.GradientTape() as tape:
        dist = compute_euclidean_distance(a_var, b_const)
    grad = tape.gradient(dist, a_var)
    assert tf.reduce_all(tf.math.is_finite(grad))
