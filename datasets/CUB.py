#Implement permuted mnist dataset for continual learning
import os

import torch
from torch.utils.data import Dataset
import torchvision
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
from tqdm.auto import tqdm

from datasets.Dataset import ContinualDataset


def get_CUB(args, num_classes_per_task=20, num_tasks=10, noise_ratio=0.0):
    assert num_classes_per_task * num_tasks <= 200, "num_classes_per_task * num_tasks must be less than or equal to 100"

    train_path = 'datasets/cub/train'
    test_path = 'datasets/cub/test'
    cache_path = 'datasets/cub/cache.torch'

    if not os.path.exists(cache_path):

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        train_dataset = torchvision.datasets.ImageFolder(train_path, transform=transform)
        test_dataset = torchvision.datasets.ImageFolder(test_path, transform=transform)

        data_train = []
        labels_train = []

        data_test = []
        labels_test = []

        print("Loading train data ...")
        for i, (img, label) in tqdm (enumerate(train_dataset)):
            data_train.append(img)
            labels_train.append(label)
            

        print("Loading test data ...")
        for i, (img, label) in tqdm (enumerate(test_dataset)):
            data_test.append(img)
            labels_test.append(label)

        labels_train = torch.tensor(labels_train)
        labels_test = torch.tensor(labels_test)

        data_train = torch.stack(data_train)
        data_test = torch.stack(data_test)

        torch.save((data_train, labels_train, data_test, labels_test), cache_path)

    else:
        print("Loading data from cache ...")
        data_train, labels_train, data_test, labels_test = torch.load(cache_path)

    if args.freeze_back:
        # precompute features

        cache_path = args.embedding_cache_path

        model = args.embedding_function

        if not os.path.exists(cache_path):
            new_data_train = []

            for d in tqdm(data_train):
                d = d.unsqueeze(0).to(args.device)
                with torch.no_grad():
                    new_data_train.append(model(d).cpu().squeeze(0))

            data_train = torch.stack(new_data_train)

            new_data_test = []

            for d in tqdm(data_test):
                d = d.unsqueeze(0).to(args.device)
                with torch.no_grad():
                    new_data_test.append(model(d).cpu().squeeze(0))

            data_test = torch.stack(new_data_test)

            torch.save((data_train, labels_train, data_test, labels_test), cache_path)
        else:
            print("Loading data from cache ...")
            data_train, labels_train, data_test, labels_test = torch.load(cache_path)

    # apply noise by randomly changing labels
    if noise_ratio > 0:
        num_noise = int(noise_ratio * len(labels_train))
        noise_idx = torch.randperm(len(labels_train))[:num_noise]
        labels_train[noise_idx] = labels_train[noise_idx][torch.randperm(num_noise)]

    data_train_ = []
    data_test_ = []

    for i in range(num_classes_per_task * num_tasks):
        d = data_train[labels_train == i]
        data_train_.append(d[torch.randperm(len(d))])
        d = data_test[labels_test == i]


        data_test_.append(d[torch.randperm(len(d))])

    data_train = data_train_
    data_test = data_test_
    del data_train_, data_test_

    data_train_ = []
    data_test_ = []

    for i in range(num_classes_per_task * num_tasks):
        data_train_.append(data_train[i])
        data_test_.append(data_test[i])


    return ContinualDataset(data_train_), ContinualDataset(data_test_)

if __name__ == '__main__':
    get_CUB()
    pass