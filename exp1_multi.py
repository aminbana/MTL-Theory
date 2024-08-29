import os
from train_continual import get_args, main
import torch
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import numpy as np
from utils import adjust_plots

args = get_args()
args.iid = 1
args.overlap = 0
args.perm = 1.0
args.samples = 0
args.fixed_mem = 0
args.lr = 1e-2
args.reset = 1
args.mem = 0
args.noise = 0.0
args.k = 64

args.dataset = 'Imagenetr'
args.dataset = 'CUB'
args.dataset = 'CIFAR100'
use_pretrained_backbone = True
args.scenario = 'task'
seeds = [101, 102, 103]
noises = [0.0, 0.1, 0.2,]
PLOT_ONLY = False


if use_pretrained_backbone:
    widths = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, ]
    widths = widths[::-1]
    l = 3
    args.freeze_back = 1
    
    if args.dataset == 'Imagenetr':
        args.backbone = 'Resnet50'
        args.load_back = 'IMAGENET1K_V2'
        
        # args.backbone = 'ViTB16'
        # args.load_back = 'vit_base_patch16_224_in21k'
                
    elif args.dataset == 'CUB':
        args.backbone = 'Resnet50'
        args.load_back = 'IMAGENET1K_V2'
        
        # args.backbone = 'ViTB16'
        # args.load_back = 'vit_base_patch16_224_in21k'
        
                
    elif args.dataset == 'CIFAR100':
        # args.backbone = 'Resnet18'
        # args.load_back = 'backbones/CIFAR10_Resnet18_Pretrain.torch'
        
        # args.backbone = 'Resnet50'
        # args.load_back = 'IMAGENET1K_V2'

        args.backbone = 'ViTB16'
        args.load_back = 'vit_base_patch16_224_in21k'
    else:
        raise NotImplementedError

    assert not (widths[-1] > 8192 and l > 3), "CUDA memory error"

else:
    args.hidden = [128] * 1
    widths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 40, 50, 64, 128]
    widths = widths[::-1]
    args.load_back = None
    args.freeze_back = 0
    args.backbone = 'Resnet18'
    

train_accs = torch.zeros (len(seeds), len(widths), 2, len(noises))
test_accs = torch.zeros (len(seeds), len(widths), 2, len(noises))

train_losses = torch.zeros (len(seeds), len(widths), 2, len(noises))
test_losses = torch.zeros (len(seeds), len(widths), 2, len(noises))

exit_ = False


for n, noise_ in enumerate (noises):
    args.noise = noise_    
    for s, seed in enumerate(seeds):
        args.seed = seed
        for j, k in enumerate(tqdm(widths)):
            if use_pretrained_backbone:
                args.hidden = [k] * l

            else:
                args.k = k
        
            for i in range(2):
                
                if i == 0:
                    if args.dataset == 'Imagenetr':
                        args.tasks = 10
                        args.classes = 20
                    elif args.dataset == 'CUB':
                        args.tasks = 10
                        args.classes = 20
                    elif args.dataset == 'CIFAR100':
                        args.tasks = 10
                        args.classes = 10

                else:
                    if args.dataset == 'Imagenetr':
                        args.tasks = 1
                        args.classes = 20
                    elif args.dataset == 'CUB':
                        args.tasks = 1
                        args.classes = 20
                    elif args.dataset == 'CIFAR100':
                        args.tasks = 1
                        args.classes = 10

                args.exp_name = (f"exp1_seed_{args.seed}_dataset_{args.dataset}_task_{args.tasks}_classes_{args.classes}"
                                f"_perm_{args.perm}_samples_{args.samples}"
                                f"_mem_{args.mem}_iid_{args.iid}_scenario_{args.scenario}_arch_{args.backbone}"
                                f"_k_{args.k}_reset_{args.reset}_noise_{args.noise}"
                                f"_hidden_{args.hidden}_loadback_{use_pretrained_backbone}_freezeback_{args.freeze_back}")

                args.save = args.exp_name
                
                if not PLOT_ONLY:
                    file_to_check = f"save/{args.save}.torch"
            
                    if os.path.isfile(file_to_check):
                        print(f"=========== =========== =========== =========== =========== {args.save} already exists")
                    else:     
                        tqdm.write(f"--------------------- Running {args.exp_name} --------------------")
                        main(args, verbose = True)

                try:
                    d = torch.load(f"save/{args.save}.torch")
                    final_epoch = d['task_epochs'][-1]

                    train_accs[s, j, i, n] = d['train_acc'][final_epoch]
                    test_accs[s, j, i, n] = d['test_acc'][final_epoch]

                    train_losses[s, j, i, n] = d['train_loss'][final_epoch]
                    test_losses[s, j, i, n] = d['test_loss'][final_epoch]
                    
                except:
                    tqdm.write(f"--------------------- {args.exp_name} not found --------------------")
                    exit_ = True
                    



if exit_ or not PLOT_ONLY:
    exit()

train_accs = train_accs.mean(0)
test_accs = test_accs.mean(0)

train_losses = train_losses.mean(0)
test_losses = test_losses.mean(0)

train_accs *= 100
test_accs *= 100

# Experiment 1 figures.

colors = ['red', 'green', 'blue', 'gray', 'orange', 'purple']
adjust_plots(font_scale=1.2)


ylim_loss_train = None
ylim_err_train = None

ylim_loss_test = None
ylim_err_test = None

if True:
    min_err = min((100 - train_accs).min(), (100 - test_accs).min())
    max_err = max((100 - train_accs).max(), (100 - test_accs).max())
    
    ylim_err_train = [min_err - 2, max_err + 2]
    ylim_err_test = ylim_err_train
        
    min_loss = min(train_losses.min(), test_losses.min())
    max_loss = max(train_losses.max(), test_losses.max())
    
    ylim_loss_train = [min_loss - 0.2, max_loss + 0.2]
    ylim_loss_test = ylim_loss_train
        

widths = np.array(widths)
if use_pretrained_backbone:
    widths = np.log2(widths)
    x_label = r'$\log_2 (k)$'
    x_ticks = widths
else:
    widths = np.log10(widths)
    x_label = r'$\log_{10} (k)$'
    x_ticks = [0, 0.5, 1, 1.5, 2,]
    

plt_name = f'{args.dataset}_{args.backbone}_{args.scenario}_{"pretrained" if use_pretrained_backbone else ""}'

####################################################################################################################################################################################################################################################################

t = -1
for i in range(2):
    for n in range(len(noises)):
        t += 1
        color = colors[t]

        label = f'multi-task, $\\sigma$ = {noises[n]}' if i == 0 else f'single-task, $\\sigma$ = {noises[n]}'

        plt.plot(widths, train_losses[:, i, n], '.-', label=f'{label}', color=color)
        plt.xlabel(x_label)
        plt.xticks(x_ticks)
        plt.ylabel('Train Loss')
        
        # if ylim_loss_train is not None:
        #     plt.ylim(ylim_loss_train)
                
        if args.scenario == 'domain':
            plt.legend()

plt.grid()
plt.savefig(f'Figs/Experiments/exp1_train_loss_{plt_name}', bbox_inches='tight', dpi=300)
plt.show()
plt.clf()

####################################################################################################################################################################################################################################################################

t = -1
for i in range(2):
    for n in range(len(noises)):
        t += 1
        color = colors[t]

        label = f'multi-task, $\\sigma$ = {noises[n]}' if i == 0 else f'single-task, $\\sigma$ = {noises[n]}'

        plt.plot(widths, test_losses[:, i, n], '.-', label=f'{label}', color=color)
        plt.xlabel(x_label)
        plt.ylabel('Test Loss')
        plt.xticks(x_ticks)
        
        # if ylim_loss_test is not None:
        #     plt.ylim(ylim_loss_test)
        
        # plt.legend()
        

plt.grid()
plt.savefig(f'Figs/Experiments/exp1_test_loss_{plt_name}', bbox_inches='tight', dpi=300)
plt.show()
plt.clf()


####################################################################################################################################################################################################################################################################


fig, axs = plt.subplots(1, 4, figsize=(28, 5))

t = -1

axs[0].grid()
axs[1].grid()
axs[2].grid()
axs[3].grid()


for i in range(2):
    for n in range(len(noises)):
        t += 1
        color = colors[t]

        label = f'multi-task, $\\sigma$ = {noises[n]}' if i == 0 else f'single-task, $\\sigma$ = {noises[n]}'


        axs[0].plot(widths, 100 - train_accs[:, i, n], '.-', label=f'{label}', color=color)
        axs[0].set_xlabel(x_label)
        axs[0].set_ylabel('Train Error')
        axs[0].legend()
        axs[0].set_xticks(x_ticks)
        if ylim_err_train is not None:
            axs[0].set_ylim(ylim_err_train)


        axs[2].plot(widths, 100 - test_accs[:, i, n], '.-', label=f'{label}', color=color)
        axs[2].set_xlabel(x_label)
        axs[2].set_ylabel('Test Error')
        axs[2].set_xticks(x_ticks)
        if ylim_err_test is not None:
            axs[2].set_ylim(ylim_err_test)


        axs[1].plot(widths, train_losses[:, i, n], '.-', label=f'{label}', color=color)
        axs[1].set_xlabel(x_label)
        axs[1].set_ylabel('Train Loss')
        axs[1].set_xticks(x_ticks)
        if ylim_loss_train is not None:
            axs[1].set_ylim(ylim_loss_train)


        axs[3].plot(widths, test_losses[:, i, n], '.-', label=f'{label}', color=color)
        axs[3].set_xlabel(x_label)
        axs[3].set_ylabel('Test Loss')
        axs[3].set_xticks(x_ticks)
        if ylim_loss_test is not None:
            axs[3].set_ylim(ylim_loss_test)


plt.savefig(f'Figs/Appendix/exp1_{plt_name}.png', bbox_inches='tight', dpi=300)
plt.clf()