"""
This script generates a dataset similar to the MultiMNIST dataset
described in [1]. However, we remove any translation.

[1] Eslami, SM Ali, et al. "Attend, infer, repeat: Fast scene
understanding with generative models." Advances in Neural Information
Processing Systems. 2016.
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import pandas as pd
import numpy as np
import numpy.random as npr
from PIL import Image
from random import shuffle
from scipy.misc import imresize

import torch
import torchvision.datasets as dset
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from utils_sgan import transform

from utils_sgan import transform


class DIGIT(Dataset):
    """Images with 0 to 4 digits of non-overlapping MNIST numbers.

    @param root: string
                 path to dataset root
    @param train: boolean [default: True]
           whether to return training examples or testing examples
    @param transform: ?torchvision.Transforms
                      optional function to apply to training inputs
    @param target_transform: ?torchvision.Transforms
                             optional function to apply to training outputs
    """
    processed_folder = 'digit'
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self, root, train=True, aug=False):
        self.root = os.path.expanduser(root)

        self.train = train  # training set or test set
        self.aug = aug

        if self.aug:
            self.input_a, self.input_b, self.class_idxA, self.class_idxB, self.label, self.per_class_n_pairs = make_dataset_augment(self.train)
        else:
            self.input_a, self.input_b, self.a_idx, self.b_idx, self.label = make_dataset_fixed(self.train)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        if self.aug:
            a_idx, b_idx = self.get_pair(index)
            a_img, b_img, label = self.input_a[a_idx], self.input_b[b_idx], self.label[index]
        else:
            a_img, b_img, label = self.input_a[self.a_idx[index]], self.input_b[self.b_idx[index]], self.label[index]


        a_img = transform(a_img, resize=32)
        b_img = transform(b_img, resize=32)

        return a_img, b_img, label, index

    def __len__(self):
        # return self.input_a.size(0)
        return len(self.label)


    def get_pair(self, index):
        label = self.label[index]
        sum = np.array(self.per_class_n_pairs).cumsum()

        if label != 0:
            within_class_index = index - sum[label-1]
        else:
            within_class_index = index
        a_idx = int(within_class_index / len(self.class_idxB[label]))
        b_idx =  within_class_index % len(self.class_idxB[label])
        a_idx = self.class_idxA[label][a_idx]
        b_idx = self.class_idxB[label][b_idx]
        assert (a_idx in self.class_idxA[label]) and b_idx in (self.class_idxB[label]), index
        return a_idx, b_idx



def load_mnist():
    train_loader = torch.utils.data.DataLoader(
        dset.MNIST(root='../data/mnist', train=True, download=True))

    test_loader = torch.utils.data.DataLoader(
        dset.MNIST(root='../data/mnist', train=False, download=True))

    train_data = {
        'imgs': train_loader.dataset.train_data.numpy(),
        'labels': train_loader.dataset.train_labels.numpy()
    }

    test_data = {
        'imgs': test_loader.dataset.test_data.numpy(),
        'labels': test_loader.dataset.test_labels.numpy()
    }
    train_class_idx = {}
    test_class_idx = {}
    for i in range(10):
        train_class_idx.update({i: []})
        test_class_idx.update({i: []})
    for i in range(len(train_data['labels'])):
        train_class_idx[train_data['labels'][i]].append(i)
    for i in range(len(test_data['labels'])):
        test_class_idx[test_data['labels'][i]].append(i)
    train_data['class_idx'] = train_class_idx
    test_data['class_idx'] = test_class_idx
    return train_data, test_data

def load_fashionMNIST():
    train_loader = torch.utils.data.DataLoader(
        dset.FashionMNIST(root='../data/fMNIST', train=True, download=True))
    test_loader = torch.utils.data.DataLoader(
        dset.FashionMNIST(root='../data/fMNIST', train=False, download=True))

    train_data = {
        'imgs': train_loader.dataset.train_data.numpy(),
        'labels': train_loader.dataset.train_labels.numpy()
    }
    test_data = {
        'imgs': test_loader.dataset.test_data.numpy(),
        'labels': test_loader.dataset.test_labels.numpy()
    }

    train_class_idx = {}
    test_class_idx = {}
    for i in range(10):
        train_class_idx.update({i: []})
        test_class_idx.update({i: []})

    new_idx = [5, 4, 0, 7, 1, 2, 6, 3, 9, 8]
    for i in range(len(train_data['labels'])):
        train_class_idx[new_idx[train_data['labels'][i]]].append(i)
    for i in range(len(test_data['labels'])):
        test_class_idx[new_idx[test_data['labels'][i]]].append(i)
    train_data['class_idx'] = train_class_idx
    test_data['class_idx'] = test_class_idx
    return train_data, test_data


def match_label(modalA, modalB):

    a_idx, b_idx, labels = [], [], []
    for i in range(len(modalA['class_idx'])):
        class_idxA = modalA['class_idx'][i]
        class_idxB = modalB['class_idx'][i]
        print('label {} len A, B: {}, {}'.format(i, len(class_idxA), len(class_idxB)))
        min_len = min(len(class_idxA), len(class_idxB))
        if len(class_idxA) > min_len:
            b_idx.extend(class_idxB)
            np.random.shuffle(class_idxA)
            a_idx.extend(class_idxA[:len(class_idxB)])
        elif len(class_idxA) < min_len:
            a_idx.extend(class_idxA)
            np.random.shuffle(class_idxB)
            b_idx.extend(class_idxB[:len(class_idxA)])
        else:
            a_idx.extend(class_idxA)
            b_idx.extend(class_idxB)
        labels.extend([i] * min_len)
    return a_idx, b_idx, labels



def make_dataset_fixed(train):
    np.random.seed(681307)
    trainA, testA = load_mnist()
    trainB, testB = load_fashionMNIST()

    if train:
        modalA = trainA
        modalB = trainB
    else:
        modalA = testA
        modalB = testB
    a_idx, b_idx, labels = match_label(modalA, modalB)

    return modalA['imgs'], modalB['imgs'], a_idx, b_idx, labels


# def make_dataset_fixed(train):
#     np.random.seed(681307)
#     trainA, testA = load_mnist()
#     trainB, testB = load_fashionMNIST()
#
#     if train:
#         input_a, input_b, labels = match_label(trainA, trainB)
#     else:
#         input_a, input_b, labels = match_label(testA, testB)
#     return input_a, input_b, labels



"""
Increase labels according to the total number of pairs.
The increased labels will be linked to each index of an pair in 'get_pair'
"""
def augment_label(modalA, modalB):
    n_labels = len(modalA['class_idx'])
    per_class_n_pairs = []
    for i in range(n_labels):
        n_a_imgs = len(modalA['class_idx'][i])
        n_b_imgs = len(modalB['class_idx'][i])
        per_class_n_pairs.append(n_a_imgs * n_b_imgs)

    all_index_pairs, all_labels = [], []
    for i in range(n_labels):
        all_labels.extend([i] * per_class_n_pairs[i])
    return per_class_n_pairs, all_labels

def make_class_idx(dataset):
    class_idx = {}

    for i in range(10):
        class_idx.update({i: []})
    for i in range(len(dataset['labels'])):
        class_idx[dataset['labels'][i]].append(i)
    return class_idx


def make_dataset_augment(train):
    np.random.seed(681307)
    trainA, testA = load_mnist()
    trainB, testB = load_fashionMNIST()

    if train:
        trainA['class_idx'] = class_idxA = make_class_idx(trainA)
        trainB['class_idx'] = class_idxB = make_class_idx(trainB)
        per_class_n_pairs, all_labels = augment_label(trainA, trainB)
        imgsA = trainA['imgs']
        imgsB = trainB['imgs']
    else:
        testA['class_idx'] = class_idxA = make_class_idx(testA)
        testB['class_idx'] = class_idxB = make_class_idx(testB)
        per_class_n_pairs, all_labels = augment_label(testA, testB)
        imgsA = testA['imgs']
        imgsB = testB['imgs']

    return imgsA, imgsB, class_idxA, class_idxB, all_labels, per_class_n_pairs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    # parser.add_argument('--min-digits', type=int, default=0,
    #                     help='minimum number of digits to add to an image')
    # parser.add_argument('--max-digits', type=int, default=4,
    #                     help='maximum number of digits to add to an image')
    # parser.add_argument('--no-resize', action='store_true', default=False,
    #                     help='if True, fix the image to be MNIST size')
    # parser.add_argument('--no-translate', action='store_true', default=False,
    #
    # args = parser.parse_args()
    # args.resize = not args.no_resize
    # args.translate = not args.no_translate
    #
    # if args.no_repeat and not args.fixed:
    #     raise Exception('Must have --fixed if --no-repeat is supplied.')
    #
    # if args.scramble and not args.fixed:
    #     raise Exception('Must have --fixed if --scramble is supplied.')
    #
    # if args.reverse and not args.fixed:
    #     raise Exception('Must have --fixed if --reverse is supplied.')
    #
    # if args.reverse and args.scramble:
    #     print('Found --reversed and --scrambling. Overriding --reversed.')
    #     args.reverse = False

    # Generate the training set and dump it to disk. (Note, this will
    # always generate the same data, else error out.)
    make_dataset_fixed('./data', 'digit', 'training.pt', 'test.pt')