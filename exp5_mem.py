import os

import torch
from tqdm.auto import tqdm

from train_continual import main, get_args
import matplotlib.pyplot as plt
import numpy as np


args = get_args()
args.tasks = 10

args.dataset = 'CIFAR100'
# args.dataset = 'Imagenetr'
# args.dataset = 'CUB'
# args.dataset = 'PMNIST'



args.iid = 0
args.overlap = 0
args.perm = 1.0
args.scenario = 'task'
args.samples = 0
args.fixed_mem = 0
args.lr = 1e-2
args.reset = 0
args.seed = 101
args.noise = 0.0
args.hidden = []
args.k = 64
args.metrics = 1

if args.dataset == 'PMNIST':
    use_pretrained_backbone = False
    mem_width_pairs = [(0, 54), (10, 48), (20, 42), (30, 34), (40, 24), (50, 1)]
    args.classes = 10
    args.backbone = 'Resnet18'
    

elif args.dataset == 'CIFAR100':
    use_pretrained_backbone = True
    mem_width_pairs = [(0, 3768), (10, 3354), (20, 2885), (30, 2329), (40, 1605), (50, 128)]
    args.classes = 10
    args.backbone = 'Resnet18'
    args.load_back = 'backbones/CIFAR10_Resnet18_Pretrain.torch'


elif args.dataset == 'CUB':
    use_pretrained_backbone = True
    mem_width_pairs = [(0, 8435), (1, 7520), (2, 6481), (3, 5249), (4, 3644), (5, 16)]
    args.classes = 20
    args.backbone = 'ViTB16'      
    args.load_back = 'vit_base_patch16_224_in21k'

elif args.dataset == 'Imagenetr':
    use_pretrained_backbone = True
    mem_width_pairs = [(0, 12028), (2, 10733), (4, 9263), (6, 7520), (8, 5249), (10, 32)]
    args.classes = 20
    args.backbone = 'ViTB16'       
    args.load_back = 'vit_base_patch16_224_in21k'
    
else:
    raise ValueError("Invalid dataset")


if use_pretrained_backbone:
    l = 3
    args.freeze_back = 1
else:
    args.hidden = [128] * 1
    args.load_back = None
    args.freeze_back = 0



seeds = [101, 102, 103]

train_accs = torch.zeros (len(seeds), len(mem_width_pairs))
test_accs = torch.zeros (len(seeds), len(mem_width_pairs))

train_losses = torch.zeros (len(seeds), len(mem_width_pairs))
test_losses = torch.zeros (len(seeds), len(mem_width_pairs))

forgetting = torch.zeros (len(seeds), len(mem_width_pairs))

for s, seed in enumerate(seeds):
    args.seed = seed
    for i, (mem,k) in tqdm (enumerate(mem_width_pairs)):
        args.mem = mem
        if use_pretrained_backbone:
            args.hidden = [k] * l
        else:
            args.k = k

        args.exp_name = (f"exp5_seed_{args.seed}_dataset_{args.dataset}_task_{args.tasks}_classes_{args.classes}"
                        f"_perm_{args.perm}_samples_{args.samples}"
                        f"_mem_{args.mem}_iid_{args.iid}_scenario_{args.scenario}_arch_{args.backbone}"
                        f"_k_{args.k}_reset_{args.reset}_noise_{args.noise}"
                        f"_hidden_{args.hidden}_loadback_{use_pretrained_backbone}_freezeback_{args.freeze_back}")

        args.save = args.exp_name
        
        tqdm.write(f"--------------------- Running {args.exp_name} --------------------")
        main(args, verbose = False)

        d = torch.load(f"save/{args.save}.torch")

        if args.iid == 0:
            assert len(d['task_epochs']) == args.tasks

        final_epoch = d['task_epochs'][-1]

        train_accs[s, i] = d['train_acc'][final_epoch]
        test_accs[s, i] = d['test_acc'][final_epoch]

        train_losses[s, i] = d['train_loss'][final_epoch]
        test_losses[s, i] = d['test_loss'][final_epoch]

        f = 0
        for t in range(args.tasks - 1):
            f += d['continual_accs'][args.tasks - 1][t] - d['continual_accs'][t][t]

        forgetting[s, i] = f / (args.tasks-1)

train_accs *= 100
test_accs *= 100

forgetting *= 100

mean_train_accs = train_accs.mean(0)
mean_test_accs = test_accs.mean(0)

mean_test_losses = test_losses.mean(0)
mean_train_losses = train_losses.mean(0)

mean_forgetting = forgetting.mean(0)

var_train_accs = train_accs.std(0)
var_test_accs = test_accs.std(0)

var_test_losses = test_losses.std(0)
var_train_losses = train_losses.std(0)

var_forgetting = forgetting.std(0)

# print max test variance
print (f"Max test variance: {var_test_accs.max()}")

print("Dataset: ", args.dataset)
for i, (mem, k) in enumerate(mem_width_pairs[::-1]):
    #train_acc: {mean_train_accs[i]:.1f} +- {var_train_accs[i]:.1f}, 
    print (f"mem: {mem}, width: {k}, test_acc: {mean_test_accs[i]:.1f} +- {var_test_accs[i]:.1f}, forgetting: {mean_forgetting[i]:.1f} +- {var_forgetting[i]:.1f}")