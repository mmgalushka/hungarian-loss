"""Steps for computing Hungarian loss."""

import tensorflow as tf

from .const import ZERO, ONE
from .ops import (
    count_zeros_in_rows,
    count_zeros_in_cols,
    get_row_mask_with_min_zeros,
    get_row_mask_with_max_zeros,
    get_col_mask_with_min_zeros,
    get_col_mask_with_max_zeros,
    expand_item_mask,
)


def compute_euclidean_distance(
    a: tf.Tensor, b: tf.Tensor
) -> tf.Tensor:  # pylint: disable=invalid-name
    """
    Computes euclidean distance between two inputs `a` and `b`.

    The computation is performed using the following formula
    `dist = sqrt((a - b)^2) = sqrt(a^2 - 2ab.T - b^2)`

    Example:
    >>> a = tf.constant(
    >>>     [[1., 2., 3., 4.],
    >>>      [5., 6., 7., 8.]]
    >>> )
    >>> b = tf.constant(
    >>>     [[1., 1., 1., 1.],
    >>>      [2., 2., 2., 2.]]
    >>> )
    >>> compute_euclidean_distance(a,b)

    >>> tf.Tensor(
    >>>     [[ 3.7416575,  2.4494898],
    >>>      [11.224972 ,  9.273619 ]] dtype=float32)

    Args:
        a:
            The 2D-tensor [num_entities, num_dimesions] of floats.
        b:
            The 2D-tensor [num_entities, num_dimesions] of floats.

    Result:
        The 2D tensor [num_entities, num_entities] with computed
        distances between entities.
    """
    a2 = tf.reshape(tf.reduce_sum(tf.square(a), axis=1), [-1, 1])
    b2 = tf.reshape(tf.reduce_sum(tf.square(b), axis=1), [1, -1])
    dist = tf.sqrt(a2 - 2 * tf.matmul(a, tf.transpose(b, perm=[1, 0])) + b2)
    return dist


def reduce_rows(matrix: tf.Tensor) -> tf.Tensor:
    """
    Subtracts the minimum value from each row.

    Example:
    >>> matrix = tf.Variable(
    >>>    [[ 30., 25., 10.],
    >>>     [ 15., 10., 20.],
    >>>     [ 25., 20., 15.]]
    >>> )
    >>> reduce_rows(matrix)

    >>> tf.Tensor(
    >>>     [[20. 15.  0.]
    >>>      [ 5.  0. 10.]
    >>>      [10.  5.  0.]], shape=(3, 3), dtype=float32)

    Args:
        matrix:
            The 2D-tensor [rows, columns] of floats to reduce.

    Returns:
        A new tensor with reduced values of the same shape as
        the input tensor.
    """
    return tf.subtract(
        matrix, tf.reshape(tf.reduce_min(matrix, axis=1), (-1, 1))
    )


def reduce_cols(matrix: tf.Tensor) -> tf.Tensor:
    """
    Subtracts the minimum value from each column.

    Example:
    >>> matrix = tf.Variable(
    >>>    [[ 30., 25., 10.],
    >>>     [ 15., 10., 20.],
    >>>     [ 25., 20., 15.]]
    >>> )
    >>> reduce_cols(matrix)

    >>> tf.Tensor(
    >>>     [[15. 15.  0.]
    >>>      [ 0.  0. 10.]
    >>>      [10. 10.  5.]], shape=(3, 3), dtype=float32)

    Args:
        matrix:
            The 2D-tensor [rows, columns] of floats to reduce.

    Returns:
        A new tensor with reduced values of the same shape as
        the input tensor.
    """
    return tf.subtract(matrix, tf.reduce_min(matrix, axis=0))


def scratch_matrix(matrix: tf.Tensor) -> tf.Tensor:
    """
    Creates the mask for rows and columns which are covering all
    zeros in the matrix.

    Example:
    >>> matrix = tf.Variable(
    >>>    [[15., 15.,  0.],
    >>>     [ 0.,  0., 10.],
    >>>     [ 5.,  5.,  0.]]
    >>> )
    >>> scratch_matrix(matrix)

    >>> (<tf.Tensor: shape=(3, 1), dtype=bool, numpy=
    >>>     array([[False],
    >>>            [ True],
    >>>            [False]])>,
    >>>  <tf.Tensor: shape=(1, 3), dtype=bool, numpy=
    >>>     array([[False, False,  True]])>)

    Args:
        matrix:
            The 2D-tensor [rows, columns] of floats to scrarch.

    Returns:
        scratched_rows_mask:
            The 2D-tensor row mask, where `True` values indicates the
            scratched rows and `False` intact rows accordingly.
        scratched_cols_mask:
            The 2D-tensor column mask, where `True` values indicates the
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

    num_of_rows, num_of_cols = matrix.shape
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


def is_optimal_assignment(
    scratched_rows_mask: tf.Tensor, scratched_cols_mask: tf.Tensor
) -> tf.Tensor:
    """
    Test if we can achieve the optimal assignment.

    We can achieve the optimal assignment if the combined number of
    scratched columns and rows equals to the matrix dimensions (since
    matrix is square, dimension side does not matter.)

    Example:

        Optimal assignment:
        >>> scratched_rows_mask = tf.constant(
        >>>    [[False], [True], [False]], tf.bool)
        >>> scratched_cols_mask = tf.constant(
        >>>    [[True, False, True]])
        >>> is_optimal_assignment(scratched_rows_mask, scratched_cols_mask)

        >>> tf.Tensor(True, shape=(), dtype=bool)

        Not optimal assignment:
        >>> scratched_rows_mask = tf.constant(
        >>>    [[False], [True], [False]], tf.bool)
        >>> scratched_cols_mask = tf.constant(
        >>>    [[True, False, True]])
        >>> is_optimal_assignment(scratched_rows_mask, scratched_cols_mask)

        >>> tf.Tensor(False, shape=(), dtype=bool)

    Args:
        scratched_rows_mask:
            The 2D-tensor row mask, where `True` values indicates the
            scratched rows and `False` intact rows accordingly.
        scratched_cols_mask:
            The 2D-tensor column mask, where `True` values indicates the
            scratched columns and `False` intact columns accordingly.

    Returns:
        The boolean tensor, where `True` indicates the optimal assignment
        and `False` otherwise.
    """
    assert scratched_rows_mask.shape[0] == scratched_cols_mask.shape[1]
    n = scratched_rows_mask.shape[0]
    number_of_lines_covering_zeros = tf.add(
        tf.reduce_sum(tf.cast(scratched_rows_mask, tf.int32)),
        tf.reduce_sum(tf.cast(scratched_cols_mask, tf.int32)),
    )
    return tf.less_equal(n, number_of_lines_covering_zeros)


def shift_zeros(matrix, scratched_rows_mask, scratched_cols_mask):
    """
    Shifts zeros in not optimal mask.

    Example:

        Optimal assignment:
        >>> matrix = tf.constant(
        >>>    [[ 30., 25., 10.],
        >>>     [ 15., 10., 20.],
        >>>     [ 25., 20., 15.]], tf.float32
        >>> )
        >>> scratched_rows_mask = tf.constant(
        >>>    [[False], [True], [False]], tf.bool)
        >>> scratched_cols_mask = tf.constant(
        >>>    [[False, False, True]])
        >>> shift_zeros(matrix, scratched_rows_mask, scratched_cols_mask)

        >>> (<tf.Tensor:
        >>>       [[10., 10.,  0.],
        >>>        [ 0.,  0., 15.],
        >>>        [ 0.,  0.,  0.]], shape=(3, 3) dtype=float32)>,
        >>> <tf.Tensor:
        >>>       [[False],
        >>>        [ True],
        >>>        [False]], shape=(3, 1), dtype=bool>,
        >>> <tf.Tensor:
        >>>       [[False, False,  True]], shape=(1, 3), dtype=bool>)

    Args:
        matrix:
            The 2D-tensor [rows, columns] of floats with reduced
            values.
        scratched_rows_mask:
            The 2D-tensor row mask, where `True` values indicates the
            scratched rows and `False` intact rows accordingly.
        scratched_cols_mask:
            The 2D-tensor column mask, where `True` values indicates the
            scratched columns and `False` intact columns accordingly.

    Returns:
        matrix:
            The 3D-tensor [rows, columns] of floats with shifted
            zeros.
        scratched_rows_mask:
            The same as input.
        scratched_cols_mask:
            The same as input
    """
    cross_mask = tf.cast(
        tf.logical_and(scratched_rows_mask, scratched_cols_mask),
        tf.float32,
    )
    inline_mask = tf.cast(
        tf.logical_or(
            tf.logical_and(
                scratched_rows_mask, tf.logical_not(scratched_cols_mask)
            ),
            tf.logical_and(
                tf.logical_not(scratched_rows_mask), scratched_cols_mask
            ),
        ),
        tf.float32,
    )
    outline_mask = tf.cast(
        tf.logical_not(
            tf.logical_or(scratched_rows_mask, scratched_cols_mask)
        ),
        tf.float32,
    )

    outline_min_value = tf.reduce_min(
        tf.math.add(
            tf.math.multiply(
                tf.math.subtract(ONE, outline_mask), tf.float32.max
            ),
            tf.math.multiply(matrix, outline_mask),
        )
    )

    cross_matrix = tf.add(
        tf.multiply(matrix, cross_mask),
        tf.multiply(outline_min_value, cross_mask),
    )
    inline_matrix = tf.multiply(matrix, inline_mask)
    outline_matrix = tf.subtract(
        tf.multiply(matrix, outline_mask),
        tf.multiply(outline_min_value, outline_mask),
    )

    return [
        tf.math.add(cross_matrix, tf.math.add(inline_matrix, outline_matrix)),
        scratched_rows_mask,
        scratched_cols_mask,
    ]


def reduce_matrix(matrix):
    """
    Reduce matrix suitable to perform the optimal assignment.

    Example:
        >>> matrix = tf.constant(
        >>>    [[ 30., 25., 10.],
        >>>     [ 15., 10., 20.],
        >>>     [ 25., 20., 15.]], tf.float32
        >>> )
        >>> reduce_matrix(matrix)

        >>> tf.Tensor(
        >>>     [[10. 10.  0.]
        >>>      [ 0.  0. 15.]
        >>>      [ 0.  0.  0.]], shape=(3, 3), dtype=float32)

    Args:
        matrix:
            The 2D-tensor [rows, columns] of floats to reduce.

    Returns:
        A new tensor representing the reduced matrix of the same
        shape as the input tensor.
    """

    def body(matrix, scratched_rows_mask, scratched_cols_mask):
        new_matrix = reduce_rows(matrix)
        new_matrix = reduce_cols(new_matrix)
        scratched_rows_mask, scratched_cols_mask = scratch_matrix(new_matrix)

        return tf.cond(
            is_optimal_assignment(scratched_rows_mask, scratched_cols_mask),
            true_fn=lambda: [
                new_matrix,
                scratched_rows_mask,
                scratched_cols_mask,
            ],
            false_fn=lambda: shift_zeros(
                new_matrix, scratched_rows_mask, scratched_cols_mask
            ),
        )

    def condition(
        matrix, scratched_rows_mask, scratched_cols_mask
    ):  # pylint: disable=unused-argument
        return tf.logical_not(
            is_optimal_assignment(scratched_rows_mask, scratched_cols_mask)
        )

    num_of_rows, num_of_cols = matrix.shape
    reduced_matrix, _, _ = tf.while_loop(
        condition,
        body,
        [
            matrix,
            tf.zeros((num_of_rows, 1), tf.bool),
            tf.zeros((1, num_of_cols), tf.bool),
        ],
    )

    return reduced_matrix


def select_optimal_assignment_mask(reduced_matrix):
    """
    Selects the optimal solution based on the reduced matrix.

    Example:
        >>> reduced_matrix = tf.constant(
        >>>     [[10. 10.  0.]
        >>>      [ 0.  0. 15.]
        >>>      [ 0.  0.  0.]], shape=(3, 3), dtype=float32)
        >>> reduce_matrix(matrix)

        >>> tf.Tensor(
        >>>     [[False False  True]
        >>>      [ True False False]
        >>>      [False  True False]], shape=(3, 3), dtype=bool)

        Args:
            matrix:
                The 2D-tensor [rows, columns] of floats representing
                the reduced matrix and used for selecting the optimal
                solution.

        Returns:
            A new tensor representing the optimal assignment mask has
            the same dimension as the input.
    """

    def select_based_on_row(zeros_mask, selection_mask):
        best_row_mask = expand_item_mask(
            get_row_mask_with_min_zeros(zeros_mask)
        )
        best_col_mask = expand_item_mask(
            get_col_mask_with_max_zeros(
                tf.logical_and(best_row_mask, zeros_mask)
            )
        )
        new_selection_mask = tf.logical_or(
            selection_mask, tf.logical_and(best_row_mask, best_col_mask)
        )
        new_mask = tf.logical_and(
            zeros_mask,
            tf.logical_not(tf.logical_or(best_row_mask, best_col_mask)),
        )
        return new_mask, new_selection_mask

    def select_based_on_col(zeros_mask, selection_mask):
        best_col_mask = expand_item_mask(
            get_col_mask_with_min_zeros(zeros_mask)
        )
        best_row_mask = expand_item_mask(
            get_row_mask_with_max_zeros(
                tf.logical_and(best_col_mask, zeros_mask)
            )
        )
        new_selection_mask = tf.logical_or(
            selection_mask, tf.logical_and(best_col_mask, best_row_mask)
        )
        new_mask = tf.logical_and(
            zeros_mask,
            tf.logical_not(tf.logical_or(best_col_mask, best_row_mask)),
        )
        return new_mask, new_selection_mask

    def body(zeros_mask, selection_mask):
        zero_count_in_rows = tf.reduce_sum(
            tf.cast(tf.equal(zeros_mask, True), tf.float32), axis=1
        )
        zero_count_in_rows = tf.where(
            tf.equal(zero_count_in_rows, ZERO),
            tf.float32.max,
            zero_count_in_rows,
        )
        min_zero_count_in_rows = tf.reduce_min(zero_count_in_rows)

        zero_count_in_cols = tf.reduce_sum(
            tf.cast(tf.equal(zeros_mask, True), tf.float32), axis=0
        )
        zero_count_in_cols = tf.where(
            tf.equal(zero_count_in_cols, ZERO),
            tf.float32.max,
            zero_count_in_cols,
        )
        min_zero_count_in_cols = tf.reduce_min(zero_count_in_cols)

        return tf.cond(
            tf.math.less(min_zero_count_in_rows, min_zero_count_in_cols),
            true_fn=lambda: select_based_on_row(zeros_mask, selection_mask),
            false_fn=lambda: select_based_on_col(zeros_mask, selection_mask),
        )

    def condition(
        zeros_mask, selection_mask
    ):  # pylint: disable=unused-argument
        return tf.reduce_any(zeros_mask)

    output = tf.while_loop(
        condition,
        body,
        [
            tf.math.equal(reduced_matrix, ZERO),
            tf.zeros(reduced_matrix.shape, tf.bool),
        ],
    )

    return output[1]
