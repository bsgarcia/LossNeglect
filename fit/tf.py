import tensorflow as tf
import numpy as np


def convert_arr_to_tensor(arr):
    arr = arr.astype(np.float32)
    arr = tf.convert_to_tensor(arr, dtype=tf.float32)
    return tf.matmul(arr, arr) + arr


def tf_minimize(
        func,
        output=None,
        cost_tol=1e-4,
        max_its=10000,
        block_size=100,
        learning_rate=1,
        device='cpu'
):
    cost = 1e10
    oldcost = cost + 2 * cost_tol
    its = 0

    with tf.Session() as sess:
        with sess.as_default():
            with tf.device('/' + device + ':0'):

                # Now create the optimizer
                optimizer = tf.train.AdamOptimizer(learning_rate).minimize(func)

                # Initialize all of the variables in the tensorflow workspace
                sess.run(tf.global_variables_initializer())

                # Now look, checking that we haven't hit the stopping conditions
                while cost < oldcost and oldcost - cost > cost_tol and its < max_its:
                    for i in range(block_size):
                        sess.run(optimizer)
                    oldcost = cost
                    cost = sess.run(func)
                its += block_size

                if output:
                    return cost, sess.run(output), its
                else:
                    return cost, its

