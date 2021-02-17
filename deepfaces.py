import torch
import torchvision.models as models
import torchvision
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize

import torch.nn as nn

import os

import cPickle as pickle


cuda_enabled = True


# a list of class names
from caffe_classes import class_names

act = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
actors = ["Alec Baldwin", "Bill Hader", "Steve Carell"]
actresses = ["Lorraine Bracco", "Peri Gilpin", "Angie Harmon"]

training_dictionary = {}
validation_dictionary = {}
test_dictionary = {}


def shuffle_images():
    for performer in act:
        if performer in actors:
            path = "cropped_rgb_227/actors"
        else:
            path = "cropped_rgb_227/actresses"
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


# We modify the torchvision implementation so that the features
# after the final pooling layer is easily accessible by calling
#       net.features(...)
# If you would like to use other layer features, you will need to
# make similar modifications.
class MyAlexNet(nn.Module):
    def load_weights(self):
        an_builtin = torchvision.models.alexnet(pretrained=True)

        features_weight_i = [0, 3, 6, 8, 10]
        for i in features_weight_i:
            self.features[i].weight = an_builtin.features[i].weight
            self.features[i].bias = an_builtin.features[i].bias

        classifier_weight_i = [1, 4, 6]
        for i in classifier_weight_i:
            self.classifier[i].weight = an_builtin.classifier[i].weight
            self.classifier[i].bias = an_builtin.classifier[i].bias

    def __init__(self, num_classes=1000):
        super(MyAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        self.load_weights()

        self.partial_features = nn.Sequential(
            *[self.features[i] for i in xrange(9)]
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

    def get_conv4(self, x):
        x = self.partial_features(x)
        return x


# model_orig = torchvision.models.alexnet(pretrained=True)
model = MyAlexNet()
model.eval()


def get_train_features(img_set):
    x_feats = np.zeros((0, 43264))
    y_feats = np.zeros((0, len(act)))

    for i, key in enumerate(act):
        last_name = key.split(' ')[1].lower()
        for a in img_set[last_name]:
            # Read an image
            if key in actors:
                im = imread('cropped_rgb_227/actors/' + a, mode='RGB')
            else:
                im = imread('cropped_rgb_227/actresses/' + a, mode='RGB')
            im = im - np.mean(im.flatten())
            im = im/np.max(np.abs(im.flatten()))
            im = np.rollaxis(im, -1).astype(np.float32)

            # turn the image into a numpy variable
            im_v = Variable(torch.from_numpy(im).unsqueeze_(0), requires_grad=False)

            # run the forward pass AlexNet prediction
            # softmax = torch.nn.Softmax()
            # all_probs = softmax(model.forward(im_v)).data.numpy()[0]
            # sorted_ans = np.argsort(all_probs)
            #
            # for i in range(-1, -6, -1):
            #     print("Answer:", class_names[sorted_ans[i]], ", Prob:", all_probs[sorted_ans[i]])
            #
            # ans = np.argmax(model.forward(im_v).data.numpy())
            # prob_ans = softmax(model.forward(im_v)).data.numpy()[0][ans]
            # print("Top Answer:", class_names[ans], "P(ans) = ", prob_ans)

            # Extract the features layer
            features = model.get_conv4(im_v).data.numpy().flatten()
            x_feats = np.vstack((x_feats, features))
            one_hot = np.zeros(len(act))
            one_hot[i] = 1
            y_feats = np.vstack((y_feats, one_hot))

    return x_feats, y_feats


def get_train():
    return get_train_features(training_dictionary)


def get_validation():
    return get_train_features(validation_dictionary)


def get_test():
    return get_train_features(test_dictionary)


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

    dim_x = 43264
    dim_h = 50
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
        pickle.dump(intermediate_perf, open("Part10Perfs.pkl", 'wb'))

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
    plt.savefig("Part10LearningCurve")


if __name__ == '__main__':
    if cuda_enabled:
        print "CUDA Enabled! Make sure you have CUDA-enabled PyTorch on your computer!"
        print "If your computer does not support CUDA, set cuda_enabled to False."
    else:
        print "CUDA Disabled! Change cuda_enabled to True to speed up training!"
    np.random.seed(0)
    shuffle_images()

    print "Grabbing features from AlexNet"
    final_model = train_model()

    intermediate_perf = pickle.load(open("Part10Perfs.pkl"))

    plot_learning_curve(intermediate_perf)
