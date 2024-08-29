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
args.scenario = 'task'
args.backbone = 'Resnet18'
args.samples = 0
args.fixed_mem = 0
args.lr = 1e-2
args.reset = 1

args.noise = 0.0
args.k = 64
args.augmentation = 0

seeds = [101, 102, 103]
ls = [0, 1, 2, 3, 4, 5]

assert use_pretrained_backbone
widths = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,][::-1]

assert args.dataset == 'CIFAR100' and args.backbone == 'Resnet18', "Pretrained backbone only available for CIFAR100 Resnet"
backbone_path = 'backbones/CIFAR10_Resnet18_Pretrain.torch'
args.load_back = backbone_path
args.freeze_back = 1

train_accs = torch.zeros (len(seeds), len(widths), 2, len(ls))
test_accs = torch.zeros (len(seeds), len(widths), 2, len(ls))

train_losses = torch.zeros (len(seeds), len(widths), 2, len(ls))
test_losses = torch.zeros (len(seeds), len(widths), 2, len(ls))

for s, seed in enumerate(seeds):
    args.seed = seed
    for i in range(1):
        for n, l in enumerate (ls):
            
            if i == 0:
                args.tasks = 10
                args.classes = 10

            else:
                args.tasks = 1
                args.classes = 10

            for j, k in enumerate(tqdm(widths)):

                args.hidden = [k] * l
                args.shared = [1] * l

                args.exp_name = (f"exp3_seed_{args.seed}_dataset_{args.dataset}_task_{args.tasks}_classes_{args.classes}"
                                f"_perm_{args.perm}_samples_{args.samples}"
                                f"_mem_{args.mem}_iid_{args.iid}_scenario_{args.scenario}_arch_{args.backbone}"
                                f"_k_{args.k}_reset_{args.reset}_noise_{args.noise}"
                                f"_hidden_{args.hidden}_loadback_{use_pretrained_backbone}_freezeback_{args.freeze_back}")

                args.save = args.exp_name
                
                file_name_to_check = f"save/{args.save}.torch"
                if os.path.exists(file_name_to_check):
                    print(f"=========== =========== =========== =========== =========== {args.save} already exists")
                else:
                    tqdm.write(f"--------------------- Running {args.exp_name} --------------------")
                    main(args, verbose = True)

                d = torch.load(f"save/{args.save}.torch")

                final_epoch = d['task_epochs'][-1]

                train_accs[s, j, i, n] = d['train_acc'][final_epoch]
                test_accs[s, j, i, n] = d['test_acc'][final_epoch]

                train_losses[s, j, i, n] = d['train_loss'][final_epoch]
                test_losses[s, j, i, n] = d['test_loss'][final_epoch]

train_accs = train_accs.mean(0)
test_accs = test_accs.mean(0)

train_losses = train_losses.mean(0)
test_losses = test_losses.mean(0)

train_accs *= 100
test_accs *= 100

# Experiment 1 figures.

import seaborn as sns
adjust_plots()

colors = sns.color_palette("tab10", n_colors=30)

widths = np.array(widths)
if use_pretrained_backbone:
    widths = np.log2(widths)
    x_label = r'$\log_2 (k)$'
    x_ticks = widths
else:
    widths = np.log10(widths)
    x_label = r'$\log_{10} (k)$'
    x_ticks = [0, 0.5, 1, 1.5, 2,]


fig, axs = plt.subplots(1, 4, figsize=(28, 5))

t = -1

axs[0].grid()
axs[1].grid()
axs[2].grid()
axs[3].grid()


max_loss = test_losses.max() + 0.1
min_loss = train_losses.min() - 0.1

for i in range(1):
    for n in range(len(ls)):
        t += 1
        color = colors[t]
        if ls[n] != 0:
            layer_str = "layers"
            if ls[n] == 1:
                layer_str = "layer"
            label = f'{ls[n]} hidden {layer_str}' if i == 0 else f'STL, {ls[n]} hidden {layer_str}'
            axs[0].plot(widths, 100 - train_accs[:, i, n], '.-', label=f'{label}', color=color)
            axs[1].plot(widths, train_losses[:, i, n], '.-', label=f'{label}', color=color)
            axs[3].plot(widths, test_losses[:, i, n], '.-', label=f'{label}', color=color)
            axs[2].plot(widths, 100 - test_accs[:, i, n], '.-', label=f'{label}', color=color)
            
        else:
            label = f'No hidden layers' if i == 0 else f'STL, no hidden layers'
            # draw a horizontal line
            axs[0].axhline(y=100 - train_accs[0, i, n], color=color, linestyle='--', label=f'{label}')
            axs[1].axhline(y=train_losses[0, i, n], color=color, linestyle='--', label=f'{label}')
            axs[3].axhline(y=test_losses[0, i, n], color=color, linestyle='--', label=f'{label}')
            axs[2].axhline(y=100 - test_accs[0, i, n], color=color, linestyle='--', label=f'{label}')
            
            
        

        



        axs[0].set_xlabel(x_label)
        axs[0].set_ylabel('Train Error')
        axs[0].legend(loc = 'upper right')
        axs[0].set_xticks(x_ticks)

        axs[2].set_xlabel(x_label)
        axs[2].set_ylabel('Test Error')
        axs[2].set_xticks(x_ticks)



        axs[1].set_xlabel(x_label)
        axs[1].set_ylabel('Train Loss')
        axs[1].set_xticks(x_ticks)

        axs[3].set_xlabel(x_label)
        axs[3].set_ylabel('Test Loss')
        axs[3].set_xticks(x_ticks)



plot_name = args.save
plt.savefig(f'Figs/Appendix/exp3_depth_{args.scenario}.png', bbox_inches='tight', dpi=300)
plt.show()
print('saved' + f'{plot_name}_multi_vs_single.png')