"""
Make test solving assignment tasks of different complexity.
"""

import tensorflow as tf
from hungarian_loss.steps import reduce_matrix, select_optimal_assignment_mask


def test_task_1():
    """Tests optimal assignment mask on task #1."""
    matrix = tf.constant(
        [[40.0, 60.0, 15.0], [25.0, 30.0, 45.0], [55.0, 30.0, 25.0]],
        tf.float16,
    )
    actual_mask = select_optimal_assignment_mask(reduce_matrix(matrix))
    expected_mask = tf.constant(
        [[False, False, True], [True, False, False], [False, True, False]],
        tf.bool,
    )
    assert tf.reduce_all(tf.equal(actual_mask, expected_mask))


def test_task_2():
    """Tests optimal assignment mask on task #2."""
    matrix = tf.constant(
        [[30.0, 25.0, 10.0], [15.0, 10.0, 20.0], [25.0, 20.0, 15.0]],
        tf.float16,
    )
    actual_mask = select_optimal_assignment_mask(reduce_matrix(matrix))
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


def test_task_3():
    """Tests optimal assignment mask on task #2."""
    matrix = tf.constant(
        [
            [38.0, 53.0, 61.0, 36.0, 66.0],
            [100.0, 60.0, 9.0, 79.0, 34.0],
            [30.0, 37.0, 36.0, 72.0, 24.0],
            [61.0, 95.0, 21.0, 14.0, 64.0],
            [89.0, 90.0, 4.0, 5.0, 79.0],
        ],
        tf.float16,
    )
    actual_mask = select_optimal_assignment_mask(reduce_matrix(matrix))
    expected_mask = tf.constant(
        [
            [True, False, False, False, False],
            [False, False, False, False, True],
            [False, True, False, False, False],
            [False, False, False, True, False],
            [False, False, True, False, False],
        ],
        tf.bool,
    )
    assert tf.reduce_all(tf.equal(actual_mask, expected_mask))
