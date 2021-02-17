import os
import random
import time
import urllib
import hashlib
import tarfile
import pickle
import gzip

from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imsave
import matplotlib.image as mpimg
from scipy.ndimage import filters
from scipy.io import loadmat


act = ['Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon', 'Alec Baldwin', 'Bill Hader', 'Steve Carell'] #list(set([a.split("\t")[0] for a in open("facescrub_actresses.txt").readlines()]))
#act = ['Kristin Chenoweth', 'Fran Drescher', 'America Ferrera', 'Daniel Radcliffe', 'Gerard Butler', 'Michael Vartan']

actors = act[3:]
actresses = act[0:3]

#subset_actors.txt


def str2int(*arg):
    retval = []
    for i in arg:
        retval.append(int(i))

    return retval


def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

testfile = urllib.URLopener()


def grab_faces(meta_file, meta=False):
    """
    Grabs faces given a metadata file
    :param meta_file: Filename
    :param meta: Whether to interpret meta_file as facescrub_<meta_file>.txt,
                 or uncropped/<meta_file>/meta.txt
    :return: None
    """
    if meta:
        meta_file = '/'.join(['uncropped_rgb', meta_file, 'meta.txt'])

    with open(meta_file, "r") as infile:
        faces = infile.readlines()

    face_type = meta_file.split('.')[0].split('_')[1]

    with open('uncropped_rgb/' + face_type + '/meta.txt', 'w+') as meta_file:

        for a in act:
            name = a.split()[1].lower()
            i = 0
            for line in faces:  # faces_subset.txt
                if a in line:
                    line_split = line.split()
                    filename = name + str(i) + '.' + line_split[4].split('.')[-1]
                    uncropped_path = "uncropped_rgb/" + face_type + '/' + filename
                    cropped_path = "cropped_rgb/" + face_type + "/" + filename

                    if not os.path.isfile(uncropped_path) or not os.path.isfile(cropped_path):
                        # A version without timeout (uncomment in case you need to
                        # unsupress exceptions, which timeout() does)
                        # testfile.retrieve(line.split()[4], "uncropped/"+filename)
                        # timeout is used to stop downloading images which take too long to download
                        timeout(testfile.retrieve, (line_split[4], uncropped_path), {}, 30)

                    if not os.path.isfile(uncropped_path):
                        print "Failed to download", line_split[4], "Skipping..."
                        continue
                    else:
                        # Do extra checking
                        with open(uncropped_path, 'rb') as infile:
                            contents = infile.read()
                            file_hash = hashlib.sha256(contents).hexdigest()
                            correct_hash = line_split[6]
                            if file_hash != correct_hash:
                                print filename + ":", "Expected:", correct_hash, "Got:", file_hash
                                with open("ded/" + file_hash, 'wb') as outfile:
                                    outfile.write(contents)
                                continue

                        print "Download Successful", filename
                        box = line_split[5].split(',')
                        x1, y1, x2, y2 = str2int(*box)
                        # crop & resize
                        try:
                            img = imread(uncropped_path, mode='RGB')
                        except IOError as e:
                            print line_split[4], "is not a valid image. Skipping..."
                            print e
                            os.remove(uncropped_path)
                            continue

                        cropped_img = img[y1:y2, x1:x2]

                        resized_img = imresize(cropped_img, (64, 64))

                        with open(cropped_path, "w") as outfile:
                            imsave(outfile, resized_img)

                        meta_file.write(line)
                        meta_file.flush()

                    i += 1

def grab_faces_unresized(meta_file, meta=False):
    """
    Grabs faces given a metadata file
    :param meta_file: Filename
    :param meta: Whether to interpret meta_file as facescrub_<meta_file>.txt,
                 or uncropped/<meta_file>/meta.txt
    :return: None
    """
    if meta:
        meta_file = '/'.join(['uncropped_rgb', meta_file, 'meta.txt'])

    with open(meta_file, "r") as infile:
        faces = infile.readlines()

    face_type = meta_file.split('.')[0].split('_')[1]

    with open('uncropped_rgb/' + face_type + '/meta.txt', 'w+') as meta_file:

        for a in act:
            name = a.split()[1].lower()
            i = 0
            for line in faces:  # faces_subset.txt
                if a in line:
                    line_split = line.split()
                    filename = name + str(i) + '.' + line_split[4].split('.')[-1]
                    uncropped_path = "uncropped_rgb/" + face_type + '/' + filename
                    cropped_path = "cropped_unresized/" + face_type + "/" + filename

                    if not os.path.isfile(uncropped_path) or not os.path.isfile(cropped_path):
                        # A version without timeout (uncomment in case you need to
                        # unsupress exceptions, which timeout() does)
                        # testfile.retrieve(line.split()[4], "uncropped/"+filename)
                        # timeout is used to stop downloading images which take too long to download
                        timeout(testfile.retrieve, (line_split[4], uncropped_path), {}, 30)

                    if not os.path.isfile(uncropped_path):
                        print "Failed to download", line_split[4], "Skipping..."
                        continue
                    else:
                        # Do extra checking
                        with open(uncropped_path, 'rb') as infile:
                            contents = infile.read()
                            file_hash = hashlib.sha256(contents).hexdigest()
                            correct_hash = line_split[6]
                            if file_hash != correct_hash:
                                print filename + ":", "Expected:", correct_hash, "Got:", file_hash
                                with open("ded/" + file_hash, 'wb') as outfile:
                                    outfile.write(contents)
                                continue

                        print "Download Successful", filename
                        box = line_split[5].split(',')
                        x1, y1, x2, y2 = str2int(*box)
                        # crop & resize
                        try:
                            img = imread(uncropped_path, mode='RGB')
                        except IOError as e:
                            print line_split[4], "is not a valid image. Skipping..."
                            print e
                            os.remove(uncropped_path)
                            continue

                        cropped_img = img[y1:y2, x1:x2]

                        with open(cropped_path, "w") as outfile:
                            imsave(outfile, cropped_img)

                        meta_file.write(line)
                        meta_file.flush()

                    i += 1


def resize_faces(size, outfolder):
    actors = os.listdir('cropped_unresized/actors')
    actresses = os.listdir('cropped_unresized/actresses')

    for a in actors:
        path = 'cropped_unresized/actors/' + a
        img = imread(path)

        resized_img = imresize(img, (size, size))

        with open(outfolder + '/actors/' + a, 'w') as outfile:
            imsave(outfile, resized_img)

    for a in actresses:
        path = 'cropped_unresized/actresses/' + a
        img = imread(path)

        resized_img = imresize(img, (size, size))

        with open(outfolder + '/actresses/' + a, 'w') as outfile:
            imsave(outfile, resized_img)


def decompress_images():
    if not os.path.isdir('faces'):
        print "Extracting Faces"
        tar = tarfile.open('faces.tar')
        tar.extractall()
        tar.close()

    if not os.path.isdir('cropped_rgb'):
        print "Extracting RGBFaces"
        tar = tarfile.open('cropped_rgb.tar')
        tar.extractall()
        tar.close()


def rgb2gray(rgb):
    '''Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''

    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray / 255.


def str2int(*arg):
    retval = []
    for i in arg:
        retval.append(int(i))

    return retval


def load_cropped_face(name, type):
    path = '/'.join(['faces', type, name])
    img = imread(path)
    if len(img.shape) > 2:
        img = rgb2gray(img)
    return img


def list_files(folder, actor=None):
    return [x for x in os.listdir('faces/{}'.format(folder)) if not actor or x.startswith(actor)]


def separate_sets(name, person_type, limit_training=70):
    """
    Returns the list of files in Training, Validation, and Test sets
    :param name: Name of actor/actress
    :param person_type: actors/actresses
    :param limit_training: Limit training set size. 0 for as large as possible.
    :return: tuple (training, validation, test)
    """
    cropped_path = '/'.join(["faces", person_type])
    items = os.listdir(cropped_path)
    all_items = [i for i in items if i.startswith(name)]

    validation = all_items[:10]
    test = all_items[10:20]
    training = all_items[20:]

    if limit_training and len(training) > limit_training:
        training = training[:limit_training]

    return training, validation, test


def make_cropped_greyscale():
    """
    Utility function to make sure that all cropped images are saved as greyscale
    :return:
    """
    infolder = 'cropped'
    outfolder = 'faces'
    actors = os.listdir('{}/actors'.format(infolder))
    actresses = os.listdir('{}/actresses'.format(infolder))

    for a in actors:
        out_a = a.split('.')[0] + '.png'
        outfile_path = '/'.join([outfolder, 'actors', out_a])
        img = imread('/'.join([infolder, 'actors', a]))
        img = rgb2gray(img)
        print "Saving", outfile_path
        imsave(outfile_path, img)

    for a in actresses:
        out_a = a.split('.')[0] + '.png'
        outfile_path = '/'.join([outfolder, 'actresses', out_a])
        img = imread('/'.join([infolder, 'actresses', a]))
        img = rgb2gray(img)
        print "Saving", outfile_path
        imsave(outfile_path, img)


def split_digits_training_set(m=None, valid=False):
    """
    Splits the training set into test and validation sets
    :param M: If this is not None, this object will be used instead of loading it from disk
    :param valid: If this is set to True, the Validation set will be returned instead of the test set
    :return: x, y
    """
    if m is None:
        m = loadmat("mnist_all.mat")

    imgs = []
    y_ground_truths = []
    for i in xrange(10):
        train_imgs = m['train{}'.format(i)]
        split = int(train_imgs.shape[0] * 0.8)
        if valid:
            to_add = train_imgs[split:]
        else:
            to_add = train_imgs[:split]

        imgs.append(to_add)
        y = [0 for _ in xrange(10)]
        y[i] = 1
        y_ground_truths = y_ground_truths + [y] * to_add.shape[0]

    x_matrix = np.vstack(imgs).T

    return x_matrix / 255.0, array(y_ground_truths)


if __name__ == "__main__":
    # Uncomment below to grab from the full list (Not used)
    # grab_faces("facescrub_actors.txt")
    # grab_faces("facescrub_actresses.txt")
    # grab_faces_unresized("facescrub_actors.txt")
    # grab_faces_unresized("facescrub_actresses.txt")

    #resize_faces(227, 'cropped_rgb_227')

    # Uncomment below to only grab the faces that were grabbed successfully
    # grab_faces("actors", True)
    # grab_faces("actresses", True)

    # name = actors[0].split()[1].lower()
    # training, validation, test = separate_sets(name, 'actors')

    # make_cropped_greyscale()

    decompress_images()
    M = loadmat("mnist_all.mat")

    selected_imgs = [list() for i in xrange(10)]

    print M

    for i in xrange(10):
        num_saved = 0
        digit = M['train{}'.format(i)]
        for ii in xrange(0, digit.shape[0], 100):
            if num_saved >= 5:
                break
            img = reshape(digit[i], (28,28))
            selected_imgs[i].append(img)
            # imsave("train{}-{}.png".format(i, ii), img)
            num_saved += 1
        plt.show()

    for i in xrange(10):
        num_saved = 0
        digit = M['test{}'.format(i)]
        for ii in xrange(0, digit.shape[0], 100):
            if num_saved >= 5:
                break
            img = reshape(digit[i], (28, 28))
            # imsave("test{}-{}.png".format(i, ii), img)
            selected_imgs[i].append(img)
            num_saved += 1
        plt.show()

    for i, digit in enumerate(selected_imgs):
        if len(digit) != 10:
            print "Insufficient images for digit {}".format(i)
            exit(1)

    for digit_num, digits in enumerate(selected_imgs):
        fig = plt.figure()
        #fig, axes = plt.subplots(nrows=5, ncols=2)#, figsize=(28, 28))

        train = digits[:5]
        test = digits[5:]

        i = 1
        for s in xrange(5):
            a = fig.add_subplot(5, 2, i)
            if i == 1:
                a.set_title('Train')
            plt.imshow(train[s])
            i += 1
            a = fig.add_subplot(5, 2, i)
            if i == 2:
                a.set_title('Test')
            plt.imshow(test[s])
            i += 1

        fig.tight_layout()
        savefig('imgs/digit-{}'.format(digit_num))
        plt.show()

    actors = list_files('actors')

    a = load_cropped_face(actors[0], 'actors')

    imshow(a)
    plt.show()
