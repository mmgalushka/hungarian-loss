"""Tests for a collection of supporting operations."""

import tensorflow as tf

from hungarian_loss.ops import (
    count_zeros_in_rows,
    count_zeros_in_cols,
    get_row_mask_with_min_zeros,
    get_row_mask_with_max_zeros,
    get_col_mask_with_min_zeros,
    get_col_mask_with_max_zeros,
    expand_item_mask,
)


def test_count_zeros_in_rows():
    """Tests the `count_zeros_in_rows` function."""
    zeros_mask = tf.constant(
        [[True, False, False], [True, True, False], [True, True, True]]
    )
    actual = count_zeros_in_rows(zeros_mask)
    expected = tf.constant([[1.0], [2.0], [3.0]], tf.float16)
    assert tf.reduce_all(tf.equal(actual, expected))


def test_count_zeros_in_cols():
    """Tests the `count_zeros_in_cols` function."""
    zeros_mask = tf.constant(
        [[True, False, False], [True, True, False], [True, True, True]]
    )
    actual = count_zeros_in_cols(zeros_mask)
    expected = tf.constant([[3.0, 2.0, 1.0]], tf.float16)
    assert tf.reduce_all(tf.equal(actual, expected))


def test_get_row_mask_with_min_zeros():
    """Tests the `get_row_mask_with_min_zeros` function."""
    zeros_mask = tf.constant(
        [[True, False, False], [True, True, False], [True, True, True]]
    )
    actual = get_row_mask_with_min_zeros(zeros_mask)
    expected = tf.constant([[True], [False], [False]], tf.bool)
    assert tf.reduce_all(tf.equal(actual, expected))

    # Tests a zeros_mask with one row containing only zeros.
    zeros_mask = tf.constant(
        [[False, False, False], [True, True, False], [True, True, True]]
    )
    actual = get_row_mask_with_min_zeros(zeros_mask)
    expected = tf.constant([[False], [True], [False]], tf.bool)
    assert tf.reduce_all(tf.equal(actual, expected))


def test_get_row_mask_with_max_zeros():
    """Tests the `get_row_mask_with_max_zeros` function."""
    zeros_mask = tf.constant(
        [[True, False, False], [True, True, False], [True, True, True]]
    )
    actual = get_row_mask_with_max_zeros(zeros_mask)
    expected = tf.constant([[False], [False], [True]], tf.bool)
    assert tf.reduce_all(tf.equal(actual, expected))


def test_get_col_mask_with_min_zeros():
    """Tests the `get_col_mask_with_min_zeros` function."""
    zeros_mask = tf.constant(
        [[True, False, False], [True, True, False], [True, True, True]]
    )
    actual = get_col_mask_with_min_zeros(zeros_mask)
    expected = tf.constant([[False, False, True]], tf.bool)
    assert tf.reduce_all(tf.equal(actual, expected))

    # Tests a zeros_mask with one column containing only zeros.
    zeros_mask = tf.constant(
        [[True, False, False], [True, True, False], [True, True, False]]
    )
    actual = get_col_mask_with_min_zeros(zeros_mask)
    expected = tf.constant([[False, True, False]], tf.bool)
    assert tf.reduce_all(tf.equal(actual, expected))


def test_get_col_mask_with_max_zeros():
    """Tests the `get_col_mask_with_max_zeros` function."""
    zeros_mask = tf.constant(
        [[True, False, False], [True, True, False], [True, True, True]]
    )
    actual = get_col_mask_with_max_zeros(zeros_mask)
    expected = tf.constant([[True, False, False]], tf.bool)
    assert tf.reduce_all(tf.equal(actual, expected))


def test_expand_item_mask():
    """Tests the `expand_item_mask` function."""
    row_mask = tf.constant([[True], [False], [False]])
    actual = expand_item_mask(row_mask)
    expected = tf.constant(
        [[True, True, True], [False, False, False], [False, False, False]]
    )
    assert tf.reduce_all(tf.equal(actual, expected))

    col_mask = tf.constant([[True, False, False]])
    actual = expand_item_mask(col_mask)
    expected = tf.constant(
        [[True, False, False], [True, False, False], [True, False, False]]
    )
    assert tf.reduce_all(tf.equal(actual, expected))
