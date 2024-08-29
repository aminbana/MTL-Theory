# Theoretical Insights into Overparameterized Models in Multi-Task and Replay-Based Continual Learning 

In this repository we provide PyTorch implementations for our paper. The directory outline is as follows:

# Installing Prerequisites

The codes in this repository are developed using the following packages:

- torch==2.0.1+cu118
- torchvision==0.15.2+cu118
- numpy==1.24.0
- timm==0.6.7

# Dataset
CIFAR-100 and MNIST datasets are downloaded automatically. For the other datasets, the user needs to download them manually and place them in the ```datasets``` directory. The datasets can be downloaded from the following links:

- [CUB-200-2011](https://www.vision.caltech.edu/datasets/cub_200_2011/)
- [ImageNet-R](https://github.com/hendrycks/imagenet-r?tab=readme-ov-file)

# Training a Single Model

To train a single model, one can use the ```train_continual.py``` script. For example, to train a continual model on the CIFAR-100 dataset using a pretrained ViTB16 backbone, one might use the following code:

```bash
python train_continual.py --dataset CIFAR100 --tasks 10 --classes 10 --backbone ViTB16  --load_back vit_base_patch16_224_in21k 
```

To train a multi-task model, just remember to turn on the iid flag:

```bash
python train_continual.py --dataset CIFAR100 --tasks 10 --classes 10 --backbone ViTB16  --load_back vit_base_patch16_224_in21k --iid 1
```

# Experiments

For easier reproducibility, we provided scripts to replicate the experiments of the paper:

- Multi-task vs single-task learning (Figure 2, 3 and 5):  ```exp1_multi.py```
- Replay-based Continual learning with different amounts of memory size (Figure 6):  ```exp4_cl.py```
- Computing the average across-task inner product of the task-optimal weights (Figure 4):  ```exp2_layers.py```
- Continual learning with fixed memory constraints (Table 1):  ```exp5_mem.py```

Simply run them as follows:

```bash
python exp1_multi.py
```


