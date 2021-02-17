from torch.autograd import Variable
import torch
import numpy as np
import matplotlib.pyplot as plt

from pylab import *
from numpy.linalg import norm
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib

from scipy.io import loadmat
import hashlib
import cPickle as pickle

import random

act = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
actors = ["Alec Baldwin", "Bill Hader", "Steve Carell"]
actresses = ["Lorraine Bracco", "Peri Gilpin", "Angie Harmon"]

training_dictionary = {}
validation_dictionary = {}
test_dictionary = {}

## Enable the following to enable CUDA support
cuda_enabled = True

def shuffle_images():
    for performer in act:
        if performer in actors:
            path = "cropped_rgb/actors"
        else:
            path = "cropped_rgb/actresses"
        actor_last_name = performer.split(" ")[1]
        image_list = []
        for filename in os.listdir(path):
            if actor_last_name.lower() in filename:
                image_list.append(filename)
        # randomize and shuffle the images
        np.random.shuffle(image_list)

        # Use the first 90 images of each actor
        # If an actor has fewer than 90 images, adjust validation and test sets accordingly
        if len(image_list) >= 100:
            validation_dictionary[actor_last_name.lower()] = image_list[:20]
            test_dictionary[actor_last_name.lower()] = image_list[20:40]
            training_dictionary[actor_last_name.lower()] = image_list[40:100]
        else:
            num_valid_test = int(20.0/100 * len(image_list))
            validation_dictionary[actor_last_name.lower()] = image_list[:num_valid_test]
            test_dictionary[actor_last_name.lower()] = image_list[num_valid_test:2 * num_valid_test]
            training_dictionary[actor_last_name.lower()] = image_list[2 * num_valid_test:]


def get_train():
    batch_xs = np.zeros((0, 64 * 64 * 3))
    batch_y_s = np.zeros((0, len(act)))

    for i, a in enumerate(act):
        a_last = a.split(" ")[1].lower()
        for j, pic in enumerate(training_dictionary[a_last]):
            if a in actors:
                filepath = 'cropped_rgb/actors/' + pic
            elif a in actresses:
                filepath = 'cropped_rgb/actresses/' + pic
            else:
                raise ValueError('File "{}" not found'.format(pic))
            img = imread(filepath, mode="RGB")
            flat_img = img.flatten()
            batch_xs = np.vstack((batch_xs, flat_img / 255.0))
            one_hot = np.zeros(len(act))
            one_hot[i] = 1
            batch_y_s = np.vstack((batch_y_s, one_hot))
    return batch_xs, batch_y_s


def get_validation():
    batch_xs = np.zeros((0, 64 * 64 * 3))
    batch_y_s = np.zeros((0, len(act)))

    for i, a in enumerate(act):
        a_last = a.split(" ")[1].lower()
        for j, pic in enumerate(validation_dictionary[a_last]):
            if a in actors:
                filepath = 'cropped_rgb/actors/' + pic
            elif a in actresses:
                filepath = 'cropped_rgb/actresses/' + pic
            else:
                raise ValueError('File "{}" not found'.format(pic))
            img = imread(filepath, mode="RGB")
            flat_img = img.flatten()
            batch_xs = np.vstack((batch_xs, flat_img / 255.0))
            one_hot = np.zeros(len(act))
            one_hot[i] = 1
            batch_y_s = np.vstack((batch_y_s, one_hot))
    return batch_xs, batch_y_s


def get_test():
    batch_xs = np.zeros((0, 64 * 64 * 3))
    batch_y_s = np.zeros((0, len(act)))

    for i, a in enumerate(act):
        a_last = a.split(" ")[1].lower()
        for j, pic in enumerate(test_dictionary[a_last]):
            if a in actors:
                filepath = 'cropped_rgb/actors/' + pic
            elif a in actresses:
                filepath = 'cropped_rgb/actresses/' + pic
            else:
                raise ValueError('File "{}" not found'.format(pic))
            img = imread(filepath, mode="RGB")
            flat_img = img.flatten()
            batch_xs = np.vstack((batch_xs, flat_img / 255.0))
            one_hot = np.zeros(len(act))
            one_hot[i] = 1
            batch_y_s = np.vstack((batch_y_s, one_hot))
    return batch_xs, batch_y_s


def train_model(save_pickle=True):
    # Actually load images
    train_x, train_y = get_train()

    valid_x, valid_y = get_validation()

    test_x, test_y = get_test()
    

    if not cuda_enabled:
        dtype_float = torch.FloatTensor
        dtype_long = torch.LongTensor
    else:
        dtype_float = torch.cuda.FloatTensor
        dtype_long = torch.cuda.LongTensor

    train_x_var = Variable(torch.from_numpy(train_x), requires_grad=False).type(dtype_float)
    valid_x_var = Variable(torch.from_numpy(valid_x), requires_grad=False).type(dtype_float)
    test_x_var = Variable(torch.from_numpy(test_x), requires_grad=False).type(dtype_float)

    train_y_classes = Variable(torch.from_numpy(np.argmax(train_y, 1)), requires_grad=False).type(dtype_long)
    valid_y_classes = Variable(torch.from_numpy(np.argmax(valid_y, 1)), requires_grad=False).type(dtype_long)
    test_y_classes = Variable(torch.from_numpy(np.argmax(test_y, 1)), requires_grad=False).type(dtype_long)

    dim_x = 64 * 64 * 3
    dim_h = 300
    dim_out = len(act)

    iter_limit = 3000
    batch_size = 32

    # Model
    if not cuda_enabled:
        model = torch.nn.Sequential(
            torch.nn.Linear(dim_x, dim_h),
            torch.nn.ReLU(),
            torch.nn.Linear(dim_h, dim_out),
        )
    else:
        model = torch.nn.Sequential(
            torch.nn.Linear(dim_x, dim_h).cuda(),
            torch.nn.ReLU().cuda(),
            torch.nn.Linear(dim_h, dim_out).cuda(),
        ).cuda()

    torch.nn.init.xavier_uniform(model[0].weight)

    loss_fn = torch.nn.CrossEntropyLoss()

    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    batches = np.random.permutation(range(train_x.shape[0]))

    intermediate_perf = {
        'train': [],
        'valid': [],
        'test': [],
    }

    print "Training NN"

    for t in range(iter_limit):
        if t % 100 == 0:
            print "epoch", t
        processed = 0
        while processed < len(batches):
            ################################################################################
            # Subsample the training set for faster training
            end = processed + batch_size if len(batches) - processed > batch_size else len(batches)
            # print "Processing [{}:{}] up to {}".format(processed, end, len(batches))
            train_idx = batches[processed: end]
            x = Variable(torch.from_numpy(train_x[train_idx]), requires_grad=False).type(dtype_float)
            y_classes = Variable(torch.from_numpy(np.argmax(train_y[train_idx], 1)), requires_grad=False).type(
                dtype_long)
            #################################################################################

            y_pred = model(x)
            loss = loss_fn(y_pred, y_classes)

            model.zero_grad()  # Zero out the previous gradient computation
            loss.backward()  # Compute the gradient
            optimizer.step()  # Use the gradient information to
            # make a step
            processed += batch_size

        if not cuda_enabled:
            y_pred_train = model(train_x_var).data.numpy()
            y_pred_valid = model(valid_x_var).data.numpy()
            y_pred_test = model(test_x_var).data.numpy()

        else:
            y_pred_train = model(train_x_var).cpu().data.numpy()
            y_pred_valid = model(valid_x_var).cpu().data.numpy()
            y_pred_test = model(test_x_var).cpu().data.numpy()

        intermediate_perf['train'].append(np.mean(np.argmax(y_pred_train, 1) == np.argmax(train_y, 1)))
        intermediate_perf['valid'].append(np.mean(np.argmax(y_pred_valid, 1) == np.argmax(valid_y, 1)))
        intermediate_perf['test'].append(np.mean(np.argmax(y_pred_test, 1) == np.argmax(test_y, 1)))

    # Make predictions using set
    x = Variable(torch.from_numpy(test_x), requires_grad=False).type(dtype_float)

    if not cuda_enabled:
        y_pred = model(x).data.numpy()
    else:
        y_pred = model(x).cpu().data.numpy()

    print np.mean(np.argmax(y_pred, 1) == np.argmax(test_y, 1))

    # pickle.dump(model, open("Part8Model.pkl", 'wb'))
    if save_pickle:
        pickle.dump(intermediate_perf, open("Part8Perfs.pkl", 'wb'))

    return model


def plot_learning_curve(learning_data):
    train_vals = learning_data["train"]
    validation_vals = learning_data["valid"]
    test_vals = learning_data["test"]

    x_axis = [i for i in range(3000)]

    fig = plt.figure()
    plt.plot(x_axis, train_vals, 'r-', label="Training Set")
    plt.plot(x_axis, validation_vals, 'y-', label="Validation Set")
    plt.plot(x_axis, test_vals, label="Test Set")

    plt.xlabel("Number of Epochs")
    plt.ylabel("Proportion of Correct Guesses")
    plt.title("Learning Curves")
    plt.legend(loc="best")
    plt.savefig("Part8LearningCurve")

def visualize_weights(weights):
    weights = weights.T

    for i, w in enumerate(weights):
        img = reshape(w, (64, 64))
        imsave("imgs/weights_visual_" + str(i) + ".png", img, cmap='RdBu')

if __name__ == "__main__":
    if cuda_enabled:
        print "CUDA Enabled! Make sure you have CUDA-enabled PyTorch installed!"
        print "If your computer does not support CUDA, set cuda_enabled to False."
    else:
        print "CUDA Disabled! Change cuda_enabled to True to speed up training!"
    np.random.seed(0)

    ## Split the images to training set, validation set and training set for part 2
    shuffle_images()

    model = train_model(save_pickle=False)
    #model = pickle.load(open('Part8Model.pkl'))

    learning_data_part_8 = pickle.load(open("Part8Perfs.pkl"))
    plot_learning_curve(learning_data_part_8)

    carell_idx = 5
    harmon_idx = 2

    in_weights = model[0].weight.cpu().data.numpy()
    out_weights = model[2].weight.cpu().data.numpy()

    carell_node = argmax(out_weights[carell_idx])
    harmon_node = argmax(out_weights[harmon_idx])

    carell_img = reshape(in_weights[carell_node], (64,64,3))
    harmon_img = reshape(in_weights[harmon_node], (64,64,3))

    imsave('Part9-Carell.png', carell_img, cmap='RdBu')
    imsave('Part9-Harmon.png', harmon_img, cmap='RdBu')
