import tensorflow as tf
import numpy as np
import time
import scipy.optimize


def convert_arr_to_tensor(arr):
    arr = arr.astype(np.float32)
    arr = tf.convert_to_tensor(arr, dtype=tf.float32)
    return tf.matmul(arr, arr) + arr


def f(a, b, c):
    s = np.ones(10) * a * b * c
    return sum(np.log(s))


def f_tensorflow(x):
    s = tf.ones(10) * x[0] * x[1] + x[2]
    return tf.reduce_sum(tf.log(s))


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
                if learning_rate:
                    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(func)
                else:
                    optimizer = tf.train.AdamOptimizer().minimize(func)

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


a = 300
b = 300
c = 300
x0 = [a, b, c]

# Scipy
func = lambda x: f(a, b, c)
startTime = time.time()
solution = scipy.optimize.minimize(func, x0)
endTime = time.time()
x = solution.x
print("  x  : ", x)
print("f(x) : ", func(x))
print(round(endTime - startTime, 4), "seconds to minimize using Scipy")
print()

# Tensorflow
X = tf.Variable(x0, dtype=tf.float32)
fx = f_tensorflow(X)


startTime = time.time()
costx, x, its = tf_minimize(fx, output=X, learning_rate=1)
endTime = time.time()
print("  x  : ", x.T)
print("f(x) : ", costx)
print(round(endTime - startTime, 4), "seconds to minimize using Tensorflow")
