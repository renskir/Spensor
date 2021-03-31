import matplotlib.pyplot as plt
import os
import argparse
import tensorflow as tf
import numpy as np
from get_data import get_data
import matplotlib
tf.enable_eager_execution()


def compute_gradients(model, inputs):
    """"
    returns gradients of a given model of a given example with respect to inputs
    (as either a numpy array or tensorflow Tensor)
    """
    X = tf.convert_to_tensor(inputs)
    with tf.GradientTape() as g:
        g.watch(X)
        y = model(X)[:, 0]
        return g.gradient(y, X)


def get_interpolated_inputs(inputs):
    """
    returns interpolated inputs from inputs
    """
    baseline = tf.zeros(shape=inputs.shape)
    alphas = tf.linspace(start=0.0, stop=1.0, num=51)  # compute baseline
    new_inputs = list()
    for j in range(alphas.shape[0]):
        new_inputs.append(baseline + (inputs - baseline) * alphas[j])

    return tf.stack(new_inputs)


def main():
    model = tf.keras.models.load_model('data/model/model')
    X, y = get_data()
    inputs = np.mean(X, axis=0)

    # get interpolated inputs
    interpolated_inputs = get_interpolated_inputs(inputs)

    # get gradients
    # sum over interpolation
    # sum over last axis (rgb)
    gradients = tf.reduce_sum(tf.reduce_sum(compute_gradients(model, interpolated_inputs), axis=0), axis=2)

    relative_gradients = gradients / tf.reduce_sum(gradients)
    relative_gradients = np.array(relative_gradients)

    matplotlib.pyplot.imshow(relative_gradients, cmap='Greys')
    matplotlib.pyplot.show()


if __name__ == '__main__':
    main()
