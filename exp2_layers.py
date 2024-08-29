import os

import torch
from tqdm.auto import tqdm

from train_continual import main, get_args
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
from utils import adjust_plots


use_pretrained_backbone = True

args = get_args()
args.dataset = 'CIFAR100'
args.iid = 1
args.overlap = 0
args.perm = 1.0
args.backbone = 'Resnet18'
args.samples = 0
args.fixed_mem = 0
args.lr = 1e-2
args.reset = 1
args.tasks = 10
args.classes = 10
args.noise = 0.0
args.k = 64

seeds = [101, 102, 103,]

assert use_pretrained_backbone
widths = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096,][::-1]
l = 3
assert not (widths[-1] > 8192 and l > 3), "CUDA memory error"
backbone_path = 'backbones/CIFAR10_Resnet18_Pretrain.torch'
args.load_back = backbone_path
args.freeze_back = 1

assert l == 3
layers = [
    ([1, 1, 1], 'domain'),
    ([0, 1, 1], 'domain'),
    ([1, 0, 1], 'domain'),
    ([1, 1, 0], 'domain'),
    ([1, 1, 1], 'task'),
    ]

train_accs = torch.zeros (len(seeds), len(widths), len(layers))
test_accs = torch.zeros (len(seeds), len(widths), len(layers))

train_losses = torch.zeros (len(seeds), len(widths), len(layers))
test_losses = torch.zeros (len(seeds), len(widths), len(layers))

for s, seed in enumerate(seeds):
    args.seed = seed
    for idx, layer in enumerate(layers):
        for j, k in enumerate(tqdm(widths)):

            args.hidden = [k] * l
            args.shared, args.scenario = layer

            args.exp_name = (f"exp2_seed_{args.seed}_dataset_{args.dataset}_task_{args.tasks}_classes_{args.classes}"
                            f"_perm_{args.perm}_samples_{args.samples}"
                            f"_iid_{args.iid}_scenario_{args.scenario}_arch_{args.backbone}"
                            f"_k_{args.k}_noise_{args.noise}"
                            f"_hidden_{args.hidden}_shared_{args.shared}_loadback_{use_pretrained_backbone}_freezeback_{args.freeze_back}")

            args.save_mlp = "output/" + args.exp_name + "/mlp.torch"

            args.save = args.exp_name
            tqdm.write(f"--------------------- Running {args.exp_name} --------------------")

            main(args, verbose = True)
