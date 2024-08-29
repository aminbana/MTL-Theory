import torch

import torch.nn as nn

from models.architectures.Resnet import ResNet18

# define memory size

# dataset = 'PMNIST'
dataset = 'CIFAR100'
# dataset = 'CUB'
# dataset = 'Imagenetr'

def get_memory_cap(m):
    if dataset == 'PMNIST':
        memory_cap_in_bytes = m * 28 * 28 * 1 * 8 * 100
    elif dataset == 'CIFAR100':
        memory_cap_in_bytes = m * 32 * 32 * 3 * 8 * 100
    else:
        memory_cap_in_bytes = m * 224 * 224 * 3 * 8 * 100
    return memory_cap_in_bytes / 1e6

# def find_m(mem_size_in_mb):
#     m = mem_size_in_mb * 1e6 // (32 * 32 * 3 * 8 * 100)
#     return m

def get_model_size(k):
    if dataset == 'PMNIST':

        model = torch.nn.Sequential(
            ResNet18(k=k, num_input_channels=1),
            nn.Linear(8 * k, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 100),
        )
    elif dataset == 'CIFAR100':
        model = torch.nn.Sequential(
            nn.Linear(512, k),
            nn.BatchNorm1d(k),
            nn.ReLU(),
            nn.Linear(k, k),
            nn.BatchNorm1d(k),
            nn.ReLU(),
            nn.Linear(k, k),
            nn.BatchNorm1d(k),
            nn.ReLU(),
            nn.Linear(k, 100),
        )
    elif dataset == 'CUB': 
        model = torch.nn.Sequential(
            nn.Linear(768, k),
            nn.BatchNorm1d(k),
            nn.ReLU(),
            nn.Linear(k, k),
            nn.BatchNorm1d(k),
            nn.ReLU(),
            nn.Linear(k, k),
            nn.BatchNorm1d(k),
            nn.ReLU(),
            nn.Linear(k, 200),
        )
    elif dataset == 'Imagenetr': 
        model = torch.nn.Sequential(
            nn.Linear(768, k),
            nn.BatchNorm1d(k),
            nn.ReLU(),
            nn.Linear(k, k),
            nn.BatchNorm1d(k),
            nn.ReLU(),
            nn.Linear(k, k),
            nn.BatchNorm1d(k),
            nn.ReLU(),
            nn.Linear(k, 200),
        )
    model_size_in_bytes = 0
    for param in model.parameters():
        model_size_in_bytes += param.data.element_size() * param.data.nelement()
    return model_size_in_bytes / 1e6

def find_k(size_in_mb, bias):
    # given size in mb apply binary search to find k

    k = 1
    while get_model_size(k + bias) < size_in_mb:
        k *= 2
    k = k // 2
    l = k
    r = 2 * k
    while l < r:
        m = (l + r) // 2
        if get_model_size(m + bias) < size_in_mb:
            l = m + 1
        else:
            r = m
    return l

if dataset == 'PMNIST':
    bias = 1
    max_mem = 50
    step = 10
elif dataset == 'CIFAR100':
    bias = 128
    max_mem = 50
    step = 10
elif dataset == 'CUB':
    bias = 16
    max_mem = 5
    step = 1
elif dataset == 'Imagenetr':
    bias = 32
    max_mem = 10
    step = 2
output = []

for m in range(0, max_mem+1, step):
    
    size_from_mem = get_memory_cap(m)
    print("m: ", m , "size: ", size_from_mem)
    k = find_k(get_memory_cap(max_mem - m), bias) + bias
    size_from_k = get_model_size(k)
    output.append((m, k))
    print (f"{m} samples: {k} size = ", f"{size_from_k} + {size_from_mem} = ", size_from_mem + size_from_k, " MB")

print (output)

print("----------------------------------")

if dataset == 'PMNIST':
    mem_width_pairs =  [(0, 54), (10, 48), (20, 42), (30, 34), (40, 24), (50, 1)]

elif dataset == 'CIFAR100':
    mem_width_pairs = [(0, 3768), (10, 3354), (20, 2885), (30, 2329), (40, 1605), (50, 128)]

elif dataset == 'CUB':
    mem_width_pairs = [(0, 8435), (1, 7520), (2, 6481), (3, 5249), (4, 3644), (5, 16)]

elif dataset == 'Imagenetr':
    mem_width_pairs = [(0, 12028), (2, 10733), (4, 9263), (6, 7520), (8, 5249), (10, 32)]

# print the ratio of memories in percentage
for mem, k in reversed (mem_width_pairs):
    space_from_k = get_model_size(k)
    space_from_mem = get_memory_cap(mem)
    print (f"{mem} samples ({space_from_mem} MB) : {k} arch ({space_from_k} MB) , ratio = ", space_from_mem / (space_from_k + space_from_mem) * 100, "%")
