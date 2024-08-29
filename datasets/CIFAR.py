#Implement permuted mnist dataset for continual learning
import os
import torch
import torchvision
from tqdm.auto import tqdm

from datasets.Dataset import ContinualDataset

def get_CIFAR100(args, num_classes_per_task = 10, num_tasks = 10, num_train = 0, num_test = 0, noise_ratio = 0.0):

    assert num_classes_per_task * num_tasks <= 100, "num_classes_per_task * num_tasks must be less than or equal to 100"

    train_dataset = torchvision.datasets.CIFAR100('./datasets/cifar100', train=True, download=True,
                                                        transform=torchvision.transforms.ToTensor())
    test_dataset = torchvision.datasets.CIFAR100('./datasets/cifar100', train=False, download=True,
                                                         transform=torchvision.transforms.ToTensor())

    data_train = torch.tensor (train_dataset.data)
    labels_train = torch.tensor (train_dataset.targets)
    data_test = torch.tensor (test_dataset.data)
    labels_test = torch.tensor (test_dataset.targets)

    data_train = data_train.float().permute([0,3,1,2]) / 255
    data_test = data_test.float().permute([0,3,1,2]) / 255

    if args.freeze_back:
        #precompute features

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

    num_train_samples = min([len(d) for d in data_train])
    num_test_samples = min([len(d) for d in data_test])

    if num_train != 0:
        assert num_train <= num_train_samples, "num_train must be less than num_train_samples"
        num_train_samples = num_train

    if num_test != 0:
        assert num_test <= num_test_samples, "num_test must be less than num_test_samples"
        num_test_samples = num_test

    data_train_ = []
    data_test_ = []

    for i in range(num_classes_per_task * num_tasks):
        data_train_.append(data_train[i][:num_train_samples])
        data_test_.append(data_test[i][:num_test_samples])

    data_train = torch.stack(data_train_)
    data_test = torch.stack(data_test_)

    return ContinualDataset(data_train), ContinualDataset(data_test)



def get_CIFAR10(args, num_classes_per_task = 2, num_tasks = 5, num_train = 0, num_test = 0, noise_ratio = 0.0):

    assert num_classes_per_task * num_tasks <= 10, "num_classes_per_task * num_tasks must be less than or equal to 100"

    train_dataset = torchvision.datasets.CIFAR10('./datasets/cifar10', train=True, download=True,
                                                        transform=torchvision.transforms.ToTensor())
    test_dataset = torchvision.datasets.CIFAR10('./datasets/cifar10', train=False, download=True,
                                                         transform=torchvision.transforms.ToTensor())

    data_train = torch.tensor (train_dataset.data)
    labels_train = torch.tensor (train_dataset.targets)
    data_test = torch.tensor (test_dataset.data)
    labels_test = torch.tensor (test_dataset.targets)

    data_train = data_train.float().permute([0,3,1,2]) / 255
    data_test = data_test.float().permute([0,3,1,2]) / 255

    if args.freeze_back:
        #precompute features

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

    num_train_samples = min([len(d) for d in data_train])
    num_test_samples = min([len(d) for d in data_test])

    if num_train != 0:
        assert num_train <= num_train_samples, "num_train must be less than num_train_samples"
        num_train_samples = num_train

    if num_test != 0:
        assert num_test <= num_test_samples, "num_test must be less than num_test_samples"
        num_test_samples = num_test

    data_train_ = []
    data_test_ = []

    for i in range(num_classes_per_task * num_tasks):
        data_train_.append(data_train[i][:num_train_samples])
        data_test_.append(data_test[i][:num_test_samples])

    data_train = torch.stack(data_train_)
    data_test = torch.stack(data_test_)

    return ContinualDataset(data_train), ContinualDataset(data_test)




if __name__ == '__main__':
    get_CIFAR100()
    pass