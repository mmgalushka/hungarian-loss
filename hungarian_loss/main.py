"""
The module implementing the Hungarian loss function.
"""

import tensorflow as tf


def euclidean_distance(a, b):  # pylint: disable=invalid-name
    """
    dist = sqrt((a - b)^2) = sqrt(a^2 - 2ab.T - b^2)
    """
    # Example of input data, both tensors have shape=(1, 2, 4):
    #
    # a = [
    #   [[1. 2. 3. 4.]
    #    [5. 6. 7. 8.]]
    # ]
    #
    # b = [
    #   [[1. 1. 1. 1.]
    #    [2. 2. 2. 2.]]
    # ]

    # N = a.shape[0]
    N = len(a)  # pylint: disable=invalid-name
    # Batch size: N = 1

    a2 = tf.reshape(  # pylint: disable=invalid-name
        tf.reduce_sum(tf.square(a), axis=2), [N, -1, 1]
    )
    # a2 = [[[ 30.]
    #        [175.]]]

    b2 = tf.reshape(  # pylint: disable=invalid-name
        tf.reduce_sum(tf.square(b), axis=2), [N, 1, -1]
    )
    # b2 = [[[4. 16.]]]

    dist = tf.sqrt(a2 - 2 * tf.matmul(a, tf.transpose(b, perm=[0, 2, 1])) + b2)
    # dist = [[[ 3.7416575  2.4494898]
    #          [11.224972   9.273619 ]]]
    return dist


def pairs_mesh(n):  # pylint: disable=invalid-name
    """Computes the pairs mesh."""
    # n == 2

    r = tf.range(n)  # pylint: disable=invalid-name
    # r = [0 1]

    a = tf.expand_dims(  # pylint: disable=invalid-name
        tf.tile(tf.expand_dims(r, 1), [1, tf.shape(r)[0]]), 2
    )
    # a = [[[0]
    #       [0]]
    #      [[1]
    #       [1]]]

    b = tf.expand_dims(  # pylint: disable=invalid-name
        tf.tile(tf.expand_dims(r, 0), [tf.shape(r)[0], 1]), 2
    )
    # b= [[[0]
    #      [1]]
    #     [[0]
    #      [1]]]

    pairs = tf.reshape(tf.concat([a, b], 2), [-1, 2])
    # pairs = [[0 0]
    #          [0 1]
    #          [1 0]
    #          [1 1]]
    return pairs


def hungarian_mask(cost):
    """Computes the Hungarian mask."""
    # cost = [[[ 3.7416575  2.4494898]
    #          [11.224972   9.273619 ]]]

    n = cost.shape[1]  # pylint: disable=invalid-name
    # n = 4

    mesh = pairs_mesh(n)
    # mesh = [[0 0]
    #         [0 1]
    #         [1 0]
    #         [1 1]]

    order = tf.argsort(tf.reshape(cost, [-1]))
    # order = [1 0 3 2]

    pairs = tf.cast(tf.gather(mesh, order, axis=None), tf.int64)
    # pairs = [[0 1]
    #          [0 0]
    #          [1 1]
    #          [1 0]]

    def body(i, pairs, mask):
        pair = tf.cast(tf.gather(pairs, indices=[i], axis=None), tf.int64)
        # pair = [0 1] -> [0 0] -> [1 1] -> [1 0]

        activation = tf.sparse.to_dense(
            tf.SparseTensor(indices=pair, values=[1], dense_shape=[n, n])
        )
        # activation = [0 1] -> [1 0] -> [0 0] -> [0 0]
        #              [0 0]    [0 0]    [0 1]    [1 0]

        probe = tf.math.add(mask, activation)
        # probe = [0 1] -> [1 1] -> [0 1] -> [0 1]
        #         [0 0]    [0 0]    [0 1]    [1 0]

        row = tf.reduce_sum(probe, axis=0)
        row = tf.where(tf.greater(row, 1))
        row = tf.equal(tf.size(row), 0)
        # row = True -> True -> False -> True

        col = tf.reduce_sum(probe, axis=1)
        col = tf.where(tf.greater(col, 1))
        col = tf.equal(tf.size(col), 0)
        # col = True -> False -> True -> True

        conjunction = tf.math.logical_and(row, col)
        # conjunction = True -> False -> False -> True

        return tf.cond(
            conjunction,
            lambda: [i + 1, pairs, probe],
            lambda: [i + 1, pairs, mask],
        )

    def condition(i, pairs, mask):  # pylint: disable=unused-argument
        return tf.less_equal(i, len(pairs) - 1)

    output = tf.while_loop(
        condition, body, [0, pairs, tf.zeros((n, n), tf.int32)]
    )
    # output = [
    #   4,
    #   [[0, 1],
    #    [0, 0],
    #    [1, 1],
    #    [1, 0]],
    #   [[0, 1],    | this is a computing
    #    [1, 0]]    | mask (second element)
    # ]
    return output[2]


def hungarian_loss(y_true, y_pred):
    """Computes the Hungarian loss"""
    v_true = tf.cast(y_true, tf.float32)
    v_pred = tf.cast(y_pred, tf.float32)

    # v_true = [
    #   [[1. 1. 1. 1.]
    #    [3. 3. 3. 3.]]
    # ]
    #
    # v_pred = [
    #   [[4. 4. 4. 4.]
    #    [2. 2. 2. 2.]]
    # ]

    dist = euclidean_distance(v_true, v_pred)
    # dist = [[[ 3.7416575  2.4494898]
    #          [11.224972   9.273619 ]]]

    mask = tf.cast(tf.map_fn(hungarian_mask, dist, tf.int32), tf.float32)
    # mask = [[[0. 1.]
    #          [1. 0.]]]

    loss = tf.reduce_sum(tf.math.multiply(dist, mask), (1, 2))
    # loss = [13.674461]

    return loss
