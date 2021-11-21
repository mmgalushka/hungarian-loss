"""A collection of supporting operations."""
import tensorflow as tf

from .const import ZERO


def count_zeros_in_rows(zeros_mask: tf.Tensor) -> tf.Tensor:
    """
    Counts a number of zero-values in each row using a zeros'-mask.

    Zeros' mask highlights the matrix cells with zero values.

    Example:
        >>> zeros_mask = tf.constant(
        >>>     [[ True, False, False],
        >>>      [ True, True, False],
        >>>      [ True, True, True]]
        >>> )
        >>> count_zeros_in_rows(zeros_mask)

        >>>  tf.Tensor(
        >>>    [[1.]
        >>>    [2.]
        >>>    [3.]], shape=(3, 1), dtype=float16)
    Args:
        zeros_mask:
            A 2D boolean tensor mask [rows, columns] for highlighting
            zero cells. The `True` value indicates that the cell in
            the original matrix is 0 and `False` otherwise.

    Returns:
        A 2D tensor representing zeros count in each row.
    """
    return tf.reshape(
        tf.reduce_sum(
            tf.cast(tf.equal(zeros_mask, True), dtype=tf.float16), axis=1
        ),
        (-1, 1),
    )


def count_zeros_in_cols(zeros_mask: tf.Tensor) -> tf.Tensor:
    """
    Counts a number of zero-values in each column using a zeros'-mask.

    Zeros' mask highlights the matrix cells with zero values.

    Example:
        >>> zeros_mask = tf.constant(
        >>>     [[ True, False, False],
        >>>      [ True, True, False],
        >>>      [ True, True, True]]
        >>> )
        >>> count_zeros_in_cols(zeros_mask)

        >>> tf.Tensor([[3. 2. 1.]], shape=(1, 3), dtype=float16)

    Args:
        zeros_mask:
            A 2D boolean tensor mask [rows, columns] for highlighting
            zero cells. The `True` value indicates that the cell in
            the original matrix is 0 and `False` otherwise.

    Returns:
        A 1D tensor representing zeros count in each column.
    """
    return tf.reshape(
        tf.reduce_sum(
            tf.cast(tf.equal(zeros_mask, True), dtype=tf.float16), axis=0
        ),
        (1, -1),
    )


def get_row_mask_with_min_zeros(zeros_mask: tf.Tensor) -> tf.Tensor:
    """
    Returns a row mask with minimum number of zeros.

    Note, rows containing all zeros are excluded from the computation
    of this mask.

    Example:

        1. Example with zeros in all rows.
        >>> zeros_mask = tf.constant(
        >>>     [[ True, False, False],
        >>>      [ True, True, False],
        >>>      [ True, True, True]]
        >>> )
        >>> get_row_mask_with_min_zeros(zeros_mask)

        >>> tf.Tensor(
        >>>     [[ True]
        >>>      [False]
        >>>      [False]], shape=(3, 1), dtype=bool)

        2. Example without zeros in one row.
        >>> zeros_mask = tf.constant(
        >>>     [[ False, False, False],
        >>>      [ True, True, False],
        >>>      [ True, True, True]]
        >>> )
        >>> get_row_mask_with_min_zeros(zeros_mask)

        >>> tf.Tensor(
        >>>     [[False]
        >>>      [ True]
        >>>      [False]], shape=(3, 1), dtype=bool)

    Args:
        zeros_mask:
            A 2D boolean tensor mask [rows, columns] for highlighting
            zero cells. The `True` value indicates that the cell in
            the original matrix is 0 and `False` otherwise.

    Returns:
        A 2D tensor represents a row mask with a minimum number of zeros.
    """
    counts = count_zeros_in_rows(zeros_mask)
    # In this step, we are replacing all zero counts with max floating
    # value, since we need this to eliminate rows filled with all zeros.
    counts = tf.where(tf.equal(counts, ZERO), tf.float16.max, counts)
    return tf.equal(
        tf.argsort(tf.argsort(counts, 0, direction="ASCENDING"), 0), 0
    )


def get_row_mask_with_max_zeros(zeros_mask: tf.Tensor) -> tf.Tensor:
    """
    Returns a row mask with maximum number of zeros.

    Example:
        >>> zeros_mask = tf.constant(
        >>>     [[ True, False, False],
        >>>      [ True, True, False],
        >>>      [ True, True, True]]
        >>> )
        >>> get_row_mask_with_max_zeros(zeros_mask)

        >>> tf.Tensor(
        >>>     [[False]
        >>>      [False]
        >>>      [ True]], shape=(3, 1), dtype=bool)
    Args:
        zeros_mask:
            A 2D boolean tensor mask [rows, columns] for highlighting
            zero cells. The `True` value indicates that the cell in
            the original matrix is 0 and `False` otherwise.

    Returns:
        A 2D tensor represents a row mask with a maximum number of zeros.
    """
    counts = count_zeros_in_rows(zeros_mask)
    return tf.equal(
        tf.argsort(tf.argsort(counts, 0, direction="DESCENDING"), 0), 0
    )


def get_col_mask_with_min_zeros(zeros_mask) -> tf.Tensor:
    """
    Returns a column mask with minimum number of zeros.

    Example:

        1. Example with zeros in all columns.
        >>> zeros_mask = tf.constant(
        >>>     [[ True, False, False],
        >>>      [ True, True, False],
        >>>      [ True, True, True]]
        >>> )
        >>> get_col_mask_with_min_zeros(zeros_mask)

        >>> tf.Tensor([[False False  True]], shape=(1, 3), dtype=bool)

        2. Example without zeros in one column.
        >>> zeros_mask = tf.constant(
        >>>     [[ True, False, False],
        >>>      [ True,  True, False],
        >>>      [ True,  True, False]]
        >>> )
        >>> get_col_mask_with_min_zeros(zeros_mask)

        >>> tf.Tensor([[False True  False]], shape=(1, 3), dtype=bool)

    Args:
        zeros_mask:
            A 2D boolean tensor mask [rows, columns] for highlighting
            zero cells. The `True` value indicates that the cell in
            the original matrix is 0 and `False` otherwise.

    Returns:
        A 2D tensor represents a column mask with a minimum number of zeros.
    """
    counts = count_zeros_in_cols(zeros_mask)
    # In this step, we are replacing all zero counts with max floating
    # value, since we need this to eliminate columns filled with all zeros.
    counts = tf.where(tf.equal(counts, ZERO), tf.float16.max, counts)
    return tf.equal(
        tf.argsort(tf.argsort(counts, 1, direction="ASCENDING"), 1), 0
    )


def get_col_mask_with_max_zeros(zeros_mask: tf.Tensor) -> tf.Tensor:
    """
    Returns a column mask with maximum number of zeros.

    Example:
        >>> zeros_mask = tf.constant(
        >>>     [[ True, False, False],
        >>>      [ True, True, False],
        >>>      [ True, True, True]]
        >>> )
        >>> get_col_mask_with_max_zeros(zeros_mask)

        >>> tf.Tensor([[ True False False]], shape=(1, 3), dtype=bool)
    Args:
        zeros_mask:
            A 2D boolean tensor mask [rows, columns] for highlighting
            zero cells. The `True` value indicates that the cell in
            the original matrix is 0 and `False` otherwise.

    Returns:
        A 2D tensor represents a column mask with a maximum number of zeros.
    """
    counts = count_zeros_in_cols(zeros_mask)
    return tf.equal(
        tf.argsort(tf.argsort(counts, 1, direction="DESCENDING"), 1), 0
    )


def expand_item_mask(item_mask: tf.Tensor) -> tf.Tensor:
    """
    Expands row or column mask to for square shape

    Example:

        1. This example of expanding row mask.
        >>> row_mask = tf.constant(
        >>>     [[ True],
        >>>      [False],
        >>>      [False]]]
        >>> )
        >>> expand_item_mask(row_mask)

        >>> tf.Tensor(
        >>>     [[ True  True  True]
        >>>      [False False False]
        >>>      [False False False]], shape=(3, 3), dtype=bool)

        2. This example of expanding column mask.
        >>> col_mask = tf.constant(
        >>>     [[ True, False, False]]]
        >>> )
        >>> expand_item_mask(col_mask)

        >>> tf.Tensor(
        >>>     [[ True False False]
        >>>      [ True False False]
        >>>      [ True False False]], shape=(3, 3), dtype=bool)

    Args:
        item_mask:
            A 2D boolean tensor mask [rows, 1] | [1, columns] for
            selected row or column.

    Returns:
        A 2D tensor [rows, columns] for the expanded row or column mask.
    """
    row_number, col_number = item_mask.get_shape()
    is_item_mask_for_rows = row_number == 1
    return tf.repeat(
        item_mask,
        repeats=(col_number if is_item_mask_for_rows else row_number),
        axis=(0 if is_item_mask_for_rows else 1),
    )
