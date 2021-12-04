"""The module implementing the Hungarian loss function."""

import tensorflow as tf

from .steps import (
    compute_euclidean_distance,
    reduce_matrix,
    select_optimal_assignment_mask,
)


class HungarianLoss(tf.keras.losses.Loss):
    """
    Computes the Hungarian loss between `y_true` and `y_pred`.

    For example, if we are detecting 10  bounding boxes on a batch
    of 32 images, the  `y_true` and `y_pred` will represent 32 images
    where each image is represented by 10 bounding boxes and each
    bounding box is represented by 4  (x,y,w,h) coordinates. This
    gives us the final shape of  `y_true` and `y_pred` `(32, 10, 4)`.

    Example:
        >>> def rmse(y_true, y_pred):
        >>>     return tf.sqrt(tf.keras.losses.mse(y_pred, y_true))
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
        >>> HungarianLoss(
        >>>    [4], 0, compute_euclidean_distance, [rmse], [1.0]
        >>> ).call(y_true, y_pred)

        >>> tf.Tensor([3.254 3.254], shape=(2,), dtype=float32)

    Args:
        slice_sizes:
            The list of slice sizes for fragmenting input `y_true` and
            `y_pred`.
        slice_index_to_compute_assignment:
            The slice index used for computing the cost matrix and performing
            the optimal assignment.
        compute_cost_matrix_fn:
            The function for computing the cost matrix.
        slice_losses_fn:
            The list of loss functions for computing loss in each slice.
        slice_weights:
            The list of weights applying to each slice loss.
    """

    def __init__(
        self,
        slice_sizes: list = None,
        slice_index_to_compute_assignment: int = None,
        compute_cost_matrix_fn: object = None,
        slice_losses_fn: list = None,
        slice_weights: list = None,
    ):
        super().__init__(
            reduction=tf.keras.losses.Reduction.NONE, name="hungarian_loss"
        )
        self.slice_sizes = slice_sizes
        self.slice_index_to_compute_assignment = (
            slice_index_to_compute_assignment
        )
        self.compute_cost_matrix_fn = compute_cost_matrix_fn
        self.slice_losses_fn = slice_losses_fn
        self.slice_weights = slice_weights

    def __compute_sample_loss(self, y_true, y_pred):  # pragma: no cover
        shift = 0
        v_trues, v_preds = [], []

        for i, size in enumerate(self.slice_sizes):
            v_true = tf.slice(
                tf.cast(y_true, tf.float32), [0, shift], [-1, size]
            )
            v_pred = tf.slice(
                tf.cast(y_pred, tf.float32), [0, shift], [-1, size]
            )

            if i == self.slice_index_to_compute_assignment:
                cost = self.compute_cost_matrix_fn(v_true, v_pred)
                # We need to reshape the distance matrix by removing the
                # `None` dimension values.
                n = cost.shape[1]
                cost = tf.reshape(cost, (n, n))

            v_trues.append(v_true)
            v_preds.append(v_pred)
            shift += size

        assignments = select_optimal_assignment_mask(reduce_matrix(cost))
        y_true_order = tf.gather(tf.where(assignments), indices=[0], axis=1)
        y_pred_order = tf.gather(tf.where(assignments), indices=[1], axis=1)

        slice_losses = []
        for loss_fn, v_true, v_pred in zip(
            self.slice_losses_fn, v_trues, v_preds
        ):
            v_true_reordered = tf.gather_nd(v_true, y_true_order)
            v_pred_reordered = tf.gather_nd(v_pred, y_pred_order)
            slice_losses.append(
                tf.reduce_mean(loss_fn(v_true_reordered, v_pred_reordered))
            )

        return tf.reduce_mean(tf.multiply(self.slice_weights, slice_losses))

    def call(self, y_true, y_pred):
        """
        The function called to compute the Hungarian.

        Args:
            y_true:
                The ground truth values, 3D `Tensor` of shape
                `[batch_size, num_of_entities, num_of_quantifiers]`.
            y_pred:
                The predicted values, 3D `Tensor` with of shape
                `[batch_size, num_of_entities, num_of_quantifiers]`.

        Returns:
            The predicted loss values 1D `tensor` with shape = `[batch_size]`.
        """
        return tf.reduce_mean(
            tf.map_fn(
                lambda x: self.__compute_sample_loss(x[0], x[1]),
                tf.cast(tf.stack([y_true, y_pred], 1), tf.float32),
            )
        )


def hungarian_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Computes the Hungarian loss between `y_true` and `y_pred`.

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

        >>> tf.Tensor([3.254 3.254], shape=(2,), dtype=float32)

    Args:
        y_true:
            The ground truth values, 3D `Tensor` of shape
            `[batch_size, num_of_entities, num_of_quantifiers]`.
        y_pred:
            The predicted values, 3D `Tensor` with of shape
            `[batch_size, num_of_entities, num_of_quantifiers]`.

    Returns:
        The predicted loss values 1D `tensor` with shape = `[batch_size]`.
    """

    def compute_sample_loss(v_true, v_pred):  # pragma: no cover
        cost = compute_euclidean_distance(v_true, v_pred)

        # We need to reshape the distance matrix by removing the
        # `None` dimension values.
        n = cost.shape[1]
        cost = tf.reshape(cost, (n, n))

        return tf.reduce_mean(
            tf.multiply(
                cost,
                tf.cast(
                    select_optimal_assignment_mask(reduce_matrix(cost)),
                    tf.float32,
                ),
            )
        )

    return tf.reduce_mean(
        tf.map_fn(
            lambda x: compute_sample_loss(x[0], x[1]),
            tf.cast(tf.stack([y_true, y_pred], 1), tf.float32),
        )
    )
