#Implement permuted mnist dataset for continual learning
import torch
from torch.utils.data import Dataset
import torchvision
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from datasets.Dataset import ContinualDataset

def prepare_mnist(num_classes, noise_ratio = 0.0):
    train_dataset = torchvision.datasets.MNIST('./datasets/mnist', train=True, download=True,
                                               transform=torchvision.transforms.ToTensor())
    test_dataset = torchvision.datasets.MNIST('./datasets/mnist', train=False, download=True,
                                              transform=torchvision.transforms.ToTensor())
    data_train = train_dataset.data
    labels_train = train_dataset.targets
    data_test = test_dataset.data
    labels_test = test_dataset.targets

    # apply noise by randomly changing labels
    if noise_ratio > 0:
        num_noise = int(noise_ratio * len(labels_train))
        noise_idx = torch.randperm(len(labels_train))[:num_noise]
        labels_train[noise_idx] = labels_train[noise_idx][torch.randperm(num_noise)]


    data_train_ = []
    data_test_ = []
    for i in range(num_classes):
        d = data_train[labels_train == i].float() / 255
        data_train_.append(d[torch.randperm(len(d))])
        d = data_test[labels_test == i].float() / 255
        data_test_.append(d[torch.randperm(len(d))])
    data_train = data_train_
    data_test = data_test_
    del data_train_, data_test_
    return data_train, data_test


def get_PermutedMNIST(args, num_classes_per_task = 10, num_tasks = 10, permute_ratio = 0.1, num_train = 0, num_test = 0, overlap_train_data = False, noise_ratio = 0.0):

    assert num_classes_per_task <= 10, "num_classes_per_task must be less than or equal to 10"

    data_train , data_test = prepare_mnist(num_classes_per_task, noise_ratio = noise_ratio)


    permutations = []
    permutations.append((torch.arange(784), torch.arange(784)))
    for i in range(num_tasks - 1):
        permutation = torch.randperm(784)[:int(784 * permute_ratio)]
        source = permutation
        destination = permutation[torch.randperm(len(permutation))]

        permutations.append((source, destination))

    num_train_samples = min([len(d) for d in data_train])

    if num_train != 0:
        if not overlap_train_data:
            assert num_train * num_tasks <= num_train_samples, "num_train * num_tasks must be less than or equal to num_train_samples"
        else:
            assert num_train <= num_train_samples, "num_train must be less than or equal to num_train_samples"
        num_train_samples = num_train
    else:
        if not overlap_train_data:
            num_train_samples = num_train_samples // num_tasks

    num_test_samples = min([len(d) for d in data_test])

    if num_test != 0:
        assert num_test <= num_test_samples, "num_test must be less than or equal to num_test_samples"
        num_test_samples = num_test


    data_train_ = []
    data_test_ = []

    for t in range(num_tasks):
        for i in range(num_classes_per_task):
            if overlap_train_data:
                d = data_train[i][:num_train_samples].clone()
            else:
                d = data_train[i][t * num_train_samples:(t + 1) * num_train_samples].clone()
            d = d.view(d.shape[0], -1)
            d[:, permutations[t - 1][1]] = d[:, permutations[t - 1][0]]
            d = d.view(d.shape[0], 28, 28)
            data_train_.append(d)

            d = data_test[i][:num_test_samples].clone()
            d = d.view(d.shape[0], -1)
            d[:, permutations[t - 1][1]] = d[:, permutations[t - 1][0]]
            d = d.view(d.shape[0], 28, 28)
            data_test_.append(d)



    data_train_ = torch.stack(data_train_, dim = 0)
    data_train_ = data_train_.unsqueeze(2)
    # print (data_train_.shape)

    data_test_ = torch.stack(data_test_, dim = 0)
    data_test_ = data_test_.unsqueeze(2)

    return ContinualDataset(data_train_), ContinualDataset(data_test_)

