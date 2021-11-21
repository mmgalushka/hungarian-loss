"""
The module implementing the Hungarian loss function.
"""

import tensorflow as tf

from .steps import (
    compute_euclidean_distance,
    reduce_matrix,
    select_optimal_assignment_mask,
)


def compute_hungarian_loss_on_distance_matrix(dist):
    """Computes the Hungarian loss on a distance matrix

    For example, if we are detecting 10 bounding boxes on an images,
    the distance matrix will represent distances between every pair
    of boxes and will have a shape `(10, 10)`.

    Example:
        >>> dist = tf.constant(
        >>>     [[ 3.742  2.45 ]
        >>>      [11.23   9.27 ]], tf.float16
        >>> )
        >>> compute_hungarian_loss_on_distance_matrix(dist)

        >>> tf.Tensor(3.254, shape=(), dtype=float16)

    Args:
        dist: The distance matrix, 2D `Tensor` of shape
            `[num_of_entities, num_of_entities]`.

    Returns:
        The loss value `Tensor` computed based on distance
        matrix.
    """
    reduced_dist = reduce_matrix(dist)
    assignment = select_optimal_assignment_mask(reduced_dist)
    return tf.reduce_mean(
        tf.multiply(dist, tf.cast(assignment, tf.float16)), (0, 1)
    )


def hungarian_loss(y_true, y_pred):
    """Computes the Hungarian loss between `y_true` and `y_pred`.

    For example, if we are detecting 10  bounding boxes on a batch
    of 32 images, the  `y_true` and `y_pred` will represent 32 images
    where each image is represented by 10 bounding boxes and each
    bounding box is represented by 4  (x,y,w,h) coordinates. This
    gives us the final shape of  `y_true` and `y_pred` `(32, 10, 4)`.

    Example:
        >>> y_true = tf.constant(
        >>>     [
        >>>         [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        >>>         [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
        >>>     ]
        >>> )
        >>> y_pred = tf.constant(
        >>>     [
        >>>         [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]],
        >>>         [[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]],
        >>>     ]
        >>> )
        >>> hungarian_loss(y_true, y_pred)

        >>> tf.Tensor([3.254 3.254], shape=(2,), dtype=float16)

    Args:
        y_true:	The ground truth values, 3D `Tensor` of shape
            `[batch_size, num_of_entities, num_of_quantifiers]`.
        y_pred:	The predicted values, 3D `Tensor` with of shape
            `[batch_size, num_of_entities, num_of_quantifiers]`.

    Returns:
        The predicted loss values 1D `tensor` with shape = `[batch_size]`.
    """
    v_true = tf.cast(y_true, tf.float32)
    v_pred = tf.cast(y_pred, tf.float32)

    dist = compute_euclidean_distance(v_true, v_pred)

    # We need to reshape the distance matrix by removing the `None`
    # dimension values.
    n = dist.shape[2]
    dist = tf.reshape(dist, (-1, n, n))

    return tf.map_fn(compute_hungarian_loss_on_distance_matrix, dist)
