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
        max_its=100000,
        block_size=8000,
        learning_rate=1,
        device='cpu'
):
    cost = [1e10, ] * 20
    oldcost = [cost[0] + 2 * cost_tol, ] * 20
    its = 0

    with tf.Session() as sess:
        with tf.device('/' + device + ':0'):

            # Now create the optimizer
            optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(func)

            # Initialize all of the variables in the tensorflow workspace
            sess.run(tf.global_variables_initializer())

            # Now look, checking that we haven't hit the stopping conditions
            while cost < oldcost and np.mean(
                    np.asarray(oldcost)[-20:] - np.asarray(cost)[-20:]) > cost_tol and its < max_its:
                for i in range(block_size):
                    sess.run(optimizer)
                oldcost.append(cost[0])
                cost.append(sess.run(func))
                its += block_size

            return cost[-1], sess.run(output), its
