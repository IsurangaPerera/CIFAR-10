import numpy as np
import torch
import torchvision

import ImageUtils


class Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.images = np.load(root_dir + '/private_test_images.npy')
        self.transform = transform

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        idx = idx.tolist() if torch.is_tensor(idx) else idx
        return self.transform(self.images[idx]) if self.transform else self.images[idx]


def load_data(data_dir):
    """Load the CIFAR-10 dataset.

    Args:
        data_dir: A string. The directory where data batches
            are stored.

    Returns:
        trainset: An torch tensor of [x_train, y_train] -> (50000,3,32,32), (50000,) 
            (dtype=np.float32)
        testset: An torch tensor of [x_test, y_test] -> (10000,3,32,32), (10000,) 
            (dtype=np.float32)
    """

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=ImageUtils.getStdImgTransformation())

    original_trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=False, transform=ImageUtils.getDefaultTransformation())

    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=False, transform=ImageUtils.getDefaultTransformation())

    return trainset, testset, original_trainset


def load_testing_images(data_dir):
    """Load the images in private testing dataset.

    Args:
        data_dir: A string. The directory where the testing images
        are stored.

    Returns:
        x_test: An numpy array of shape [N, 32, 32, 3].
            (dtype=np.float32)
    """
    return np.load('../data/private_test_images.npy')


def train_valid_split(trainset, original_trainset, train_ratio=0.8):
    """Split the original training data into a new training dataset
    and a validation dataset.

    Args:
        trainset: An torch tensor of [x_train, y_train] -> (50000,3,32,32), (50000,) 
            (dtype=np.float32)
        original_trainset: An torch tensor of [x_train, y_train] -> (50000,3,32,32), (50000,)
            (dtype=np.float32)
        train_ratio: A float number between 0 and 1.

    Returns:
        TODO
    """
    if train_ratio == 1:
        return trainset, None

    train_size = int(train_ratio * len(trainset))
    valid_size = len(trainset) - train_size
    train, _ = torch.utils.data.random_split(trainset, [train_size, valid_size])
    _, valid = torch.utils.data.random_split(original_trainset, [train_size, valid_size])

    return train, valid
