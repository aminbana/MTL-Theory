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
args.tasks = 10
args.dataset = 'CIFAR100'
args.iid = 0
args.overlap = 0
args.perm = 1.0
args.scenario = 'task'
args.backbone = 'Resnet18'
args.samples = 0
args.classes = 10
args.fixed_mem = 0
args.lr = 1e-2
args.reset = 0
args.seed = 101
args.noise = 0.0
args.hidden = []
args.k = 64
args.metrics = 1


seeds = [101, 102, 103,]
mems = [0, 5, 100, 300, 500]

PLOT_ONLY = True

if use_pretrained_backbone:
    widths_ = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]
    widths = sorted (widths_ + [w+w//2 for w in widths_[1:-1]])
    l = 3
    assert not (widths[-1] > 8192 and l > 3), "CUDA memory error"
    backbone_path = 'backbones/CIFAR10_Resnet18_Pretrain.torch'
    
    args.load_back = backbone_path
    args.freeze_back = 1
else:
    args.hidden = [128] * 1
    widths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 64, 128]   
    args.load_back = None
    args.freeze_back = 0


train_accs = torch.zeros (len(seeds), len(widths), len(mems))
test_accs = torch.zeros (len(seeds), len(widths), len(mems))

train_losses = torch.zeros (len(seeds), len(widths), len(mems))
test_losses = torch.zeros (len(seeds), len(widths), len(mems))

forgetting = torch.zeros (len(seeds), len(widths), len(mems))


exit_ = False

for s, seed in enumerate(seeds):
    args.seed = seed
    for i, mem in enumerate(mems):
        args.mem = mem
        for j, k in enumerate(tqdm(widths)):
            if use_pretrained_backbone:

                args.hidden = [k] * l

            else:
                args.k = k

            args.exp_name = (f"exp4_seed_{args.seed}_dataset_{args.dataset}_task_{args.tasks}_classes_{args.classes}"
                            f"_perm_{args.perm}_samples_{args.samples}"
                            f"_mem_{args.mem}_iid_{args.iid}_scenario_{args.scenario}_arch_{args.backbone}"
                            f"_k_{args.k}_reset_{args.reset}_noise_{args.noise}"
                            f"_hidden_{args.hidden}_loadback_{use_pretrained_backbone}_freezeback_{args.freeze_back}")

            args.save = args.exp_name
            
            file_to_check = f"save/{args.save}.torch"
            
            if not PLOT_ONLY: 
                if os.path.isfile(file_to_check):
                    print(f"=========== =========== =========== =========== =========== {args.save} already exists")
                else:
                    tqdm.write(f"--------------------- Running {args.exp_name} --------------------")
                    main(args, verbose = False)

            try:
                d = torch.load(f"save/{args.save}.torch")

                if args.iid == 0:
                    assert len (d['task_epochs']) == args.tasks

                final_epoch = d['task_epochs'][-1]

                train_accs[s, j, i] = d['train_acc'][final_epoch]
                test_accs[s, j, i] = d['test_acc'][final_epoch]

                train_losses[s, j, i] = d['train_loss'][final_epoch]
                test_losses[s, j, i] = d['test_loss'][final_epoch]

                f = 0

                for t in range(args.tasks - 1):
                    f += d['continual_losses'][args.tasks - 1][t] - d['continual_losses'][t][t]

                forgetting[s, j, i] = f / (args.tasks-1)
            except:
                print(f"{args.save} not found")
                exit_ = True

if exit_ or not PLOT_ONLY:
    exit()

train_accs = train_accs.mean(0)
test_accs = test_accs.mean(0)

test_losses = test_losses.mean(0)
train_losses = train_losses.mean(0)

forgetting = forgetting.mean(0)


train_accs *= 100
test_accs *= 100

# Experiment 4 figures.

colors = ['red', 'green', 'blue', 'gray', 'orange', 'purple', 'darkblue']
adjust_plots()

widths = np.array(widths)
if use_pretrained_backbone:
    widths = np.log2(widths)
    x_label = r'$\log_2 (k)$'
    x_ticks = np.log2(widths_)
else:
    widths = np.log10(widths)
    x_label = r'$\log_{10} (k)$'
    x_ticks = [0, 0.5, 1, 1.5, 2,]


plot_name = f'{args.scenario}_{"pretrained" if use_pretrained_backbone else ""}'

################################################################################################################################################################

t = -1

for i in range(len(mems)):
        t += 1
        color = colors[t]

        label = rf'$m$ = {mems[i]}'
        # if i == 0:
        #     label += ' (sequential fine-tuning)'
        # if i == len(mems) - 1:
        #     label += ' (multi-task)'

        plt.plot(widths, train_losses[:, i], '.-', label=label, color=color)
        plt.xlabel(x_label)
        plt.ylabel('Train Loss')
        plt.legend()

plt.grid()
# plt.savefig(f'Figs/Experiments/exp4_train_loss_{plot_name}.png', bbox_inches='tight', dpi = 300)
plt.show()
plt.clf()

################################################################################################################################################################

t = -1

for i in range(len(mems)):
        t += 1
        color = colors[t]

        label = rf'$m$ = {mems[i]}'
        # if i == 0:
        #     label += ' (sequential fine-tuning)'
        # if i == len(mems) - 1:
        #     label += ' (multi-task)'

        plt.plot(widths, test_losses[:, i], '.-', label=label, color=color)

        plt.xlabel(x_label)
        plt.ylabel('Test Loss')
        plt.xticks(x_ticks)
        if use_pretrained_backbone:
            plt.legend()
        
plt.grid()
plt.savefig(f'Figs/Experiments/exp4_test_loss_{plot_name}.png', bbox_inches='tight', dpi = 300)
plt.show()
plt.clf()

################################################################################################################################################################

t = -1

for i in range(len(mems)):
        t += 1
        color = colors[t]

        label = rf'$m$ = {mems[i]}'
        # if i == 0:
        #     label += ' (sequential)'
        # if i == len(mems) - 1:
        #     label += ' (multi-task)'

        plt.plot(widths, forgetting[:, i], '.-', label=f'{label}', color=color)
        plt.xlabel(x_label)
        plt.ylabel('Forgetting')
        # plt.legend()
        plt.xticks(x_ticks)


plt.grid()
plt.savefig(f'Figs/Experiments/exp4_forgetting_{plot_name}.png', bbox_inches='tight', dpi = 300)
plt.show()
plt.clf()

################################################################################################################################################################

fig, axs = plt.subplots(1, 4, figsize=(28, 5))

t = -1

axs[0].grid()
axs[1].grid()
axs[2].grid()
axs[3].grid()

for i in range(len(mems)):
        t += 1
        color = colors[t]

        label = rf'$m$ = {mems[i]}'
        if i == 0:
            label += ' (sequential fine-tuning)'
        if i == len(mems) - 1:
            label += ' (multi-task)'

        axs[0].plot(widths, train_losses[:, i], '.-', label=f'{label}', color=color)
        axs[0].set_xlabel(x_label)
        axs[0].set_ylabel('Last Task Train Error')
        axs[0].legend()
        axs[0].set_xticks(x_ticks)

        axs[2].plot(widths, 100 - test_accs[:, i], '.-', label=f'{label}', color=color)
        axs[2].set_xlabel(x_label)
        axs[2].set_ylabel('Test Error')
        axs[2].set_xticks(x_ticks)
        

        axs[1].plot(widths, forgetting[:, i], '.-', label=f'{label}', color=color)
        axs[1].set_xlabel(x_label)
        axs[1].set_ylabel('Forgetting')
        axs[1].set_xticks(x_ticks)

        axs[3].plot(widths, test_losses[:, i], '.-', label=f'{label}', color=color)
        axs[3].set_xlabel(x_label)
        axs[3].set_ylabel('Test Loss')
        axs[3].set_xticks(x_ticks)

plt.savefig(f'Figs/Appendix/exp4_{plot_name}.png', bbox_inches='tight', dpi=300)
plt.show()




