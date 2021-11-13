"""
Steps for computing Hungarian loss.
"""

import tensorflow as tf

from . import ZERO
from .ops import (
    count_zeros_in_rows,
    count_zeros_in_cols,
    get_row_mask_with_max_zeros,
    get_col_mask_with_max_zeros,
    expand_item_mask,
)


def reduce_rows(matrix: tf.Tensor) -> tf.Tensor:
    """Subtracts the minimum value from each row.

    Example:
    >>> matrix = tf.Variable(
    >>>    [[[ 30., 25., 10.],
    >>>      [ 15., 10., 20.],
    >>>      [ 25., 20., 15.]]]
    >>> )
    >>> reduce_rows(matrix)

    >>> tf.Tensor(
    >>>     [[[20. 15.  0.]
    >>>       [ 5.  0. 10.]
    >>>       [10.  5.  0.]]], shape=(1, 3, 3), dtype=float16)

    Args:
        matrix:
            The 3D [batch, rows, columns] of floats to reduce.

    Returns:
        A new tensor with reduced values of the same shape as
        the input tensor.
    """
    return tf.cast(
        tf.subtract(
            matrix, tf.reshape(tf.reduce_min(matrix, axis=2), (-1, 1))
        ),
        tf.float16,
    )


def reduce_cols(matrix):
    """Subtracts the minimum value from each column.

    Example:
    >>> matrix = tf.Variable(
    >>>    [[[ 30., 25., 10.],
    >>>      [ 15., 10., 20.],
    >>>      [ 25., 20., 15.]]]
    >>> )
    >>> reduce_cols(matrix)

    >>> tf.Tensor(
    >>>     [[[15. 15.  0.]
    >>>       [ 0.  0. 10.]
    >>>       [10. 10.  5.]]], shape=(1, 3, 3), dtype=float16)

    Args:
        matrix:
            The 3D [batch, rows, columns] of floats to reduce.

    Returns:
        A new tensor with reduced values of the same shape as
        the input tensor.
    """
    return tf.cast(
        tf.subtract(matrix, tf.reduce_min(matrix, axis=1)), tf.float16
    )


def scratch_matrix(matrix):
    """Creates the mask for rows and columns which are covering all
    zeros in the matrix.

    Example:
    >>> matrix = tf.Variable(
    >>>    [[[15., 15.,  0.],
    >>>      [ 0.,  0., 10.],
    >>>      [ 5.,  5.,  0.]]]
    >>> )
    >>> scratch_row(matrix)

    >>> (<tf.Tensor: shape=(3, 1), dtype=bool, numpy=
    >>>     array([[False],
    >>>            [ True],
    >>>            [False]])>,
    >>>  <tf.Tensor: shape=(1, 3), dtype=bool, numpy=
    >>>     array([[False, False,  True]])>)

    Args:
        matrix:
            The 3D [batch, rows, columns] of floats to scrarch.

    Returns:
        scratched_rows_mask:
            The 2D row mask, where `True` values indicates the
            scratched rows and `False` intact rows accordingly.
        scratched_cols_mask:
            The 2D column mask, where `True` values indicates the
            scratched columns and `False` intact columns accordingly.
    """

    def scratch_row(zeros_mask, scratched_rows_mask, scratched_cols_mask):
        scratched_row_mask = get_row_mask_with_max_zeros(zeros_mask)
        new_scratched_rows_mask = tf.logical_or(
            scratched_rows_mask, scratched_row_mask
        )
        new_zeros_mask = tf.logical_and(
            zeros_mask, tf.logical_not(expand_item_mask(scratched_row_mask))
        )
        return new_zeros_mask, new_scratched_rows_mask, scratched_cols_mask

    def scratch_col(zeros_mask, scratched_rows_mask, scratched_cols_mask):
        scratched_col_mask = get_col_mask_with_max_zeros(zeros_mask)
        new_scratched_cols_mask = tf.logical_or(
            scratched_cols_mask, scratched_col_mask
        )
        new_zeros_mask = tf.logical_and(
            zeros_mask, tf.logical_not(expand_item_mask(scratched_col_mask))
        )
        return new_zeros_mask, scratched_rows_mask, new_scratched_cols_mask

    def body(zeros_mask, scratched_rows_mask, scratched_cols_mask):
        return tf.cond(
            tf.math.greater(
                tf.reduce_max(count_zeros_in_rows(zeros_mask)),
                tf.reduce_max(count_zeros_in_cols(zeros_mask)),
            ),
            true_fn=lambda: scratch_row(
                zeros_mask, scratched_rows_mask, scratched_cols_mask
            ),
            false_fn=lambda: scratch_col(
                zeros_mask, scratched_rows_mask, scratched_cols_mask
            ),
        )

    def condition(zeros_mask, scratched_rows_mask, scratched_cols_mask):
        return tf.reduce_any(zeros_mask)

    _, num_of_rows, num_of_cols = matrix.shape
    _, scratched_rows_mask, scratched_cols_mask = tf.while_loop(
        condition,
        body,
        [
            tf.math.equal(matrix, ZERO),
            tf.zeros((num_of_rows, 1), tf.bool),
            tf.zeros((1, num_of_cols), tf.bool),
        ],
    )

    return scratched_rows_mask, scratched_cols_mask
