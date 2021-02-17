from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
from scipy.ndimage import filters
import urllib
from numpy import random

import cPickle

import os
from scipy.io import loadmat

import get_data

np.seterr(all='raise')

#Load the MNIST digit data
M = loadmat("mnist_all.mat")

#Display the 150-th "5" digit from the training set
# imshow(M["train5"][150].reshape((28,28)), cmap=cm.gray)
# show()


def softmax(o):
    '''Return the output of the softmax function for the matrix of output o. o
    is an NxM matrix where N is the number of outputs for a single case, and M
    is the number of cases'''
    return exp(o)/tile(sum(exp(o),0), (len(o),1))

def tanh_layer(y, W, b):
    '''Return the output of a tanh layer for the input matrix y. y
    is an NxM matrix where N is the number of inputs for a single case, and M
    is the number of cases'''
    return tanh(dot(W.T, y)+b)

def forward(x, W0, b0, W1, b1):
    L0 = tanh_layer(x, W0, b0)
    L1 = dot(W1.T, L0) + b1
    output = softmax(L1)
    return L0, L1, output

def NLL(y, y_):
    """Cost function, y is the softmax w.r.t outputs, y_ is a vector of ground truths"""
    return -sum(y_*log(y))

def deriv_multilayer(W0, b0, W1, b1, x, L0, L1, y, y_):
    '''Incomplete function for computing the gradient of the cross-entropy
    cost function w.r.t the parameters of a neural network'''
    dCdL1 =  y - y_
    dCdW1 =  dot(L0, dCdL1.T )

def compute_output(weights, x_matrix, b):
    """
    Returns the linear combination of x_matrix matrix and weights matrix plus b the bias unit vector  (part 2)
    (output vector (o))
    """
    return np.dot(weights.T, x_matrix) + b

def compute_gradient(p_i, y_i, x_matrix):
    """ number of categories = 10
    p_i (10 x m) is softmax, y_i (10 x m) is the ground truth, x_matrix (784 x m) is the matrix of pixels of images
    The resulting matrix should be 10 x 784"""
    return np.dot(p_i - y_i, x_matrix.T)

def cost_f(weights, x_matrix, b, y):
    """Returns the cost function of the neural network"""
    return NLL(softmax(compute_output(weights, x_matrix, b)), y)


def finite_difference(h=0.000000000001, initial_weights_coefficient=0, m=10):
    """weights are 784 x 10, x_matrix is the 784 x 10 image input matrix
    (number of digits, we are trying to classify from 0-9),
    ground truth is 10 x 10 where m is number of sample images.

    Return the gradient matrix, the finite difference matrix and the difference matrix between them
    """
    # Build up the x_matrix -> 784 x m and the y ground truths
    # Get 10 images
    imgs = []
    y_ground_truths = np.identity(10)
    for i in xrange(10):
        img = M['train{}'.format(i)][0]
        imgs.append(img)

    x_matrix = np.vstack(imgs).T / 255.0

    b = np.ones((10, 1))
    #b = np.zeros(10)

    # Build up the weights matrix
    initial_weights = []
    for i in range(x_matrix.shape[0]):
        initial_weights_row = initial_weights_coefficient * np.ones(10)
        initial_weights.append(initial_weights_row)
    initial_weights = np.vstack(initial_weights)
    initial_weights_copy = initial_weights.copy()

    # gradient dimension is 10 x 784
    gradient = compute_gradient(softmax(compute_output(initial_weights, x_matrix, b)), y_ground_truths, x_matrix)
    lst_finite_difference = []

    # lst_finite_difference dimension is 10 x 784
    for row_idx in range(784):
        lst_row = []
        for column_idx in range(10):
            cost_function_original = cost_f(initial_weights, x_matrix, b, y_ground_truths)
            initial_weights_copy[row_idx][column_idx] = initial_weights_copy[row_idx][column_idx] + h
            cost_function_added_h = cost_f(initial_weights_copy, x_matrix, b, y_ground_truths)
            finite_difference = (cost_function_added_h - cost_function_original) / h
            lst_row.append(finite_difference)
            initial_weights_copy = initial_weights.copy()
        lst_finite_difference.append(lst_row)
    return gradient, lst_finite_difference, gradient - array(lst_finite_difference).T


def df(x, y, t, b):
    o = compute_output(t, x, b)
    p_i = softmax(o)
    return compute_gradient(p_i, y, x).T


def grad_descent(f, df, x, y, init_t, alpha, b, max_iter=10000):
    """
    Computes weights via gradient descent
    :param f: Cost function
    :param df: Gradient of the cost function
    :param x: Inputs
    :param y: Ground Truth
    :param init_t: Initial Weights
    :param alpha: Learning Rate
    :param max_iter: Max number of iterations
    :return: Vector/Matrix of weights
    """
    intermediate_weights = {}
    EPS = 1e-3   #EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    iter = 0
    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        t -= alpha*df(x, y, t, b)
        if iter % 200 == 0:
            print "Iter", iter
            # print "x = (%.2f, %.2f, %.2f), f(x) = %.2f" % (t[0], t[1], t[2], f(x, y, t))
            print "Gradient: ", df(x, y, t, b), "\n"

        if iter % 200 == 0:
            intermediate_weights[iter] = t.copy()
        iter += 1
    return t, intermediate_weights


def train_neural_net(initial_weights_coefficient=1e-3, alpha=1e-6, max_iter=10000):
    """Returns the trained weights of the neuro network"""
    # Build up the x_matrix -> 784 x m and the y ground truths
    # Get 10 images
    x_matrix, y_ground_truths = get_data.split_digits_training_set(M)

    b = np.ones((10, 1))

    # Build up the weights matrix
    initial_weights = []
    for i in range(x_matrix.shape[0]):
        initial_weights_row = initial_weights_coefficient * np.ones(10)
        initial_weights.append(initial_weights_row)
    initial_weights = np.vstack(initial_weights)

    result = grad_descent(cost_f, df, x_matrix, y_ground_truths.T, initial_weights, alpha, b, max_iter)

    return result


def get_performance(trained_weights):
    x, y = get_data.split_digits_training_set(M, valid=True)
    b = np.ones((10, 1))
    output_array = compute_output(trained_weights, x, b)
    print "Shape of the output array", output_array.shape
    softmax_array = softmax(output_array)
    output_index_list = np.argmax(output_array, 0)
    y_index_list = np.argmax(y, 1)

    correct = 0
    print y.shape, softmax_array.shape
    for i in range(x.shape[1]):
        if output_index_list[i] == y_index_list[i]:
            correct += 1
    print float(correct) / x.shape[1]



def plot_learning_curve(weights, part_number):
    x_train, y_train = get_data.split_digits_training_set(M)
    x_valid, y_valid = get_data.split_digits_training_set(M, valid=True)
    b = np.ones((10, 1))

    x_vals = []
    y_vals_train = []
    y_vals_valid = []

    num_iters = weights.keys()
    num_iters.sort()

    for w in num_iters:
        cost_train = cost_f(weights[w], x_train, b, y_train.T) / x_train.shape[1]
        cost_val = cost_f(weights[w], x_valid, b, y_valid.T) / x_valid.shape[1]

        # w is the x axis, cost is the y axis
        x_vals.append(w)
        y_vals_train.append(cost_train)
        y_vals_valid.append(cost_val)

    fig = plt.figure()
    plt.plot(x_vals, y_vals_train, 'r-', label="Training Set")
    plt.plot(x_vals, y_vals_valid, label="Validation Set")

    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost")
    plt.title("Average Cost vs. Number of Iterations")
    plt.legend(loc="best")
    plt.savefig("part_" + str(part_number) + "_validation_training_plot")


def visualize_weights(weights):
    weights = weights.T

    for i, w in enumerate(weights):
        img = reshape(w, (28,28))
        imsave("imgs/weights_visual_" + str(i) + ".png", img, cmap='RdBu')

# Part 5
def grad_descent_momentum(f, df, x, y, init_t, alpha, b, max_iter=10000, momentum=0.99):
    """
    Computes weights via gradient descent
    :param f: Cost function
    :param df: Gradient of the cost function
    :param x: Inputs
    :param y: Ground Truth
    :param init_t: Initial Weights
    :param alpha: Learning Rate
    :param max_iter: Max number of iterations
    :return: Vector/Matrix of weights
    """
    intermediate_weights = {}
    EPS = 1e-2   #EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    iter = 0
    v = 0
    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        v = (momentum * v) + alpha*df(x, y, t, b)
        t = t - v
        if iter % 200 == 0:
            print "Iter", iter
            # print "x = (%.2f, %.2f, %.2f), f(x) = %.2f" % (t[0], t[1], t[2], f(x, y, t))
            print "Gradient: ", df(x, y, t, b), "\n"

        if iter % 200 == 0:
            intermediate_weights[iter] = t.copy()
        iter += 1
    return t, intermediate_weights

def train_neural_net_momentum(initial_weights_coefficient=1e-3, alpha=1e-6, max_iter=10000):
    """Returns the trained weights of the neuro network"""
    # Build up the x_matrix -> 784 x m and the y ground truths
    # Get 10 images
    x_matrix, y_ground_truths = get_data.split_digits_training_set(M)

    b = np.ones((10, 1))

    # Build up the weights matrix
    initial_weights = []
    for i in range(x_matrix.shape[0]):
        initial_weights_row = initial_weights_coefficient * np.ones(10)
        initial_weights.append(initial_weights_row)
    initial_weights = np.vstack(initial_weights)

    result = grad_descent_momentum(cost_f, df, x_matrix, y_ground_truths.T, initial_weights, alpha, b, max_iter)

    return result

# Part 6
def grad_descent_momentum_two_weights(f, df, x, y, init_t, alpha, b, w1, w2, max_iter=100, momentum=0.99):
    """
    Computes weights via gradient descent
    :param f: Cost function
    :param df: Gradient of the cost function
    :param x: Inputs
    :param y: Ground Truth
    :param init_t: Initial Weights
    :param alpha: Learning Rate
    :param max_iter: Max number of iterations
    :return: Vector/Matrix of weights
    """
    intermediate_weights = {}
    EPS = 1e-5  # EPS = 10**(-5)
    prev_t = init_t - 10 * EPS
    t = init_t.copy()
    iter = 0
    v = 0
    while norm(t - prev_t) > EPS and iter < max_iter:# and abs(t[500][4] - w1) > EPS and abs(t[500][5] - w2) > EPS:
        prev_t = t.copy()
        v = (momentum * v) + alpha * df(x, y, t, b)
        t[500][4] -= v[500][4]
        t[500][5] -= v[500][5]
        if iter % 200 == 0:
            print "Iter", iter
            # print "x = (%.2f, %.2f, %.2f), f(x) = %.2f" % (t[0], t[1], t[2], f(x, y, t))
            print "Gradient: ", df(x, y, t, b), "\n"

        intermediate_weights[iter] = t.copy()
        iter += 1
    return t, intermediate_weights

def train_neural_net_two_weights(initial_weights, w1, w2, initial_weights_coefficient=1e-3, alpha=1e-6, max_iter=100):
    """Returns the trained weights of the neuro network"""
    # Build up the x_matrix -> 784 x m and the y ground truths
    # Get 10 images
    x_matrix, y_ground_truths = get_data.split_digits_training_set(M)

    b = np.ones((10, 1))

    result = grad_descent_momentum_two_weights(cost_f, df, x_matrix, y_ground_truths.T, initial_weights, alpha, b, w1, w2, max_iter, 0)

    return result

def train_neural_net_momentum_two_weights(initial_weights, w1, w2, initial_weights_coefficient=1e-3, alpha=1e-6, max_iter=100):
    """Returns the trained weights of the neuro network"""
    # Build up the x_matrix -> 784 x m and the y ground truths
    # Get 10 images
    x_matrix, y_ground_truths = get_data.split_digits_training_set(M)

    b = np.ones((10, 1))

    result = grad_descent_momentum_two_weights(cost_f, df, x_matrix, y_ground_truths.T, initial_weights, alpha, b, w1, w2, max_iter)

    return result

def compute_cost_function_part_6a(result_weights):
    """0-2 range x1 and x2, indices are (4, 500) for x1 and (5, 500) for x2"""
    weights = result_weights

    x_axis_array = np.arange(-2.0, 2.0, 0.2)
    x_axis_lst = x_axis_array.tolist()
    y_axis_lst = x_axis_lst

    x_train, y_train = get_data.split_digits_training_set(M)
    x_valid, y_valid = get_data.split_digits_training_set(M, valid=True)
    b = np.ones((10, 1))

    X, Y = np.meshgrid(x_axis_lst, y_axis_lst)
    Z_train = array(X)
    Z_valid = array(X)

    w1_orig = weights[500][4]
    w2_orig = weights[500][5]

    for i in xrange(len(x_axis_lst)):
        for j in xrange(len(y_axis_lst)):
            weights[500][4] = x_axis_lst[i] + w1_orig
            weights[500][5] = y_axis_lst[j] + w2_orig
            cost_train = cost_f(weights, x_train, b, y_train.T)
            cost_valid = cost_f(weights, x_valid, b, y_valid.T)
            Z_train[i,j] = cost_train
            Z_valid[i,j] = cost_valid

    fig = plt.figure()
    plt.contour(X, Y, Z_train)
    plt.xlabel("w1 values")
    plt.ylabel("w2 values")
    plt.title("Contour plot of x1 and x2")
    plt.legend(loc="best")
    plt.savefig("part_6a_contour_plot")


def plot_trajectory(gd_data, mo_data, filename="part6_contour_plot_with_trajectory", plt_gd=True, plt_mo=True):
    weights = gd_data[0]

    indices = gd_data[1].keys()
    indices.sort()

    gd_traj = [(gd_data[1][i][500][4], gd_data[1][i][500][5]) for i in indices]
    mo_traj = [(mo_data[1][i][500][4], mo_data[1][i][500][5]) for i in indices]

    x_axis_array = np.arange(-2.0, 2.0, 0.2)
    x_axis_lst = x_axis_array.tolist()
    y_axis_lst = x_axis_lst

    x_train, y_train = get_data.split_digits_training_set(M)
    x_valid, y_valid = get_data.split_digits_training_set(M, valid=True)
    b = np.ones((10, 1))

    X, Y = np.meshgrid(x_axis_lst, y_axis_lst)
    Z_train = array(X)
    Z_valid = array(X)

    w1_orig = weights[500][4]
    w2_orig = weights[500][5]

    for i in xrange(len(x_axis_lst)):
        for j in xrange(len(y_axis_lst)):
            weights[500][4] = x_axis_lst[i] + w1_orig
            weights[500][5] = y_axis_lst[j] + w2_orig
            cost_train = cost_f(weights, x_train, b, y_train.T) / x_train.shape[1]
            cost_valid = cost_f(weights, x_valid, b, y_valid.T) / x_train.shape[1]
            Z_train[i, j] = cost_train
            Z_valid[i, j] = cost_valid
            # z_axis_lst_train.append(cost_train)
            # z_axis_lst_valid.append(cost_valid)

    fig = plt.figure()
    CS = plt.contour(X, Y, Z_train)
    plt.xlabel("w1 values")
    plt.ylabel("w2 values")
    plt.title("Contour plot of x1 and x2")
    plt.clabel(CS, inline=1, fontsize=10)
    if plt_gd:
        plt.plot([a for a, b in gd_traj], [b for a, b in gd_traj], 'yo-', label="No Momentum")
    if plt_mo:
        plt.plot([a for a, b in mo_traj], [b for a, b in mo_traj], 'go-', label="Momentum")
    plt.legend(loc="best")
    plt.savefig(filename)


if __name__ == '__main__':

    #Load sample weights for the multilayer neural network
    snapshot = cPickle.load(open("snapshot50.pkl"))
    W0 = snapshot["W0"]
    b0 = snapshot["b0"].reshape((300,1))
    W1 = snapshot["W1"]
    b1 = snapshot["b1"].reshape((10,1))

    ##Load one example from the training set, and run it through the
    ##neural network
    x = M["train5"][148:149].T
    L0, L1, output = forward(x, W0, b0, W1, b1)
    ## get the index at which the output is the largest
    y = argmax(output)

    ## Part 3b
    gradient, lst_finite_difference, difference = finite_difference()

    # Part 4
    #result_weights = cPickle.load(open("part4-20k.pkl"))
    # result_weights = cPickle.load(open("part4_2.pkl"))
    result_weights = train_neural_net(max_iter=10000)
    cPickle.dump(result_weights, open("part4_2.pkl", 'wb'))
    get_performance(result_weights[0])
    visualize_weights(result_weights[0])
    plot_learning_curve(result_weights[1], 4)

    # part 5
    # result_weights_momentum = cPickle.load(open("part5_2.pkl"))
    result_weights_momentum = train_neural_net_momentum(max_iter=10000)
    cPickle.dump(result_weights_momentum, open("part5_2.pkl", 'wb'))
    plot_learning_curve(result_weights_momentum[1], 5)

    # part 6a
    result_two_weights_momentum = cPickle.load(open("part5_2.pkl"))[0]
    # compute_cost_function_part_6a(result_two_weights_momentum)

    # part 6c
    result_weights_part6b = cPickle.load(open("part5_2.pkl"))[0]
    w1, w2 = result_weights_part6b[500][4:6]
    result_weights_part6b[500][4] = -2
    result_weights_part6b[500][5] = -2
    p6b_weights = train_neural_net_two_weights(result_weights_part6b, w1, w2, alpha=1e-2)
    result_weights_part6b[500][4] = -2
    result_weights_part6b[500][5] = -2
    p6c_weights_099 = train_neural_net_momentum_two_weights(result_weights_part6b, w1, w2, alpha=1e-3)
    cPickle.dump(p6b_weights, open("part6b_nomomentum.pkl", 'wb'))
    cPickle.dump(p6c_weights_099, open("part6c-099momentum.pkl", 'wb'))
    gd_data = cPickle.load(open("part6b_nomomentum.pkl"))
    mo_data = cPickle.load(open("part6c-099momentum.pkl"))
    plot_trajectory(gd_data, mo_data)
    plot_trajectory(gd_data, mo_data, plt_gd=False, filename="part6_contour_plot_with_trajectory-099momentum")
    plot_trajectory(gd_data, mo_data, plt_mo=False, filename="part6_contour_plot_with_trajectory-nomomentum")



    ################################################################################
    #Code for displaying a feature from the weight matrix mW
    #fig = figure(1)
    #ax = fig.gca()
    #heatmap = ax.imshow(mW[:,50].reshape((28,28)), cmap = cm.coolwarm)
    #fig.colorbar(heatmap, shrink = 0.5, aspect=5)
    #show()
    ################################################################################
