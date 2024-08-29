import argparse
import os
from datetime import datetime
from pathlib import Path
import torch

from datasets.CIFAR import get_CIFAR100, get_CIFAR10
from datasets.MNIST import get_PermutedMNIST
from datasets.Imagenetr import get_Imagenetr
from datasets.CUB import get_CUB

from tqdm.auto import tqdm
from continual_utils import Continual_Util

from models.neural import DNN
from utils import set_seed, get_gpu_with_max_free_memory


def get_args(run_in_notebook = False):
    parser = argparse.ArgumentParser()

    # Input options
    parser.add_argument('--hidden', type=int, nargs='*', default=[], help="Hidden sizes of the classification FCs")
    parser.add_argument('--shared', type=int, nargs='*', default=[], help="Use shared or unshared FCs")
    parser.add_argument('--backbone', default='Resnet18', choices=['Resnet18', 'Resnet50', 'ViTB16'], help="Optimizer")
    parser.add_argument('--k', default=64, type=int, help="The model parameter size")

    parser.add_argument('--lr', default=1e-2, type=float, help="Learning rate")
    parser.add_argument('--bs', default=512, type=int, help="Batch size")
    parser.add_argument('--optim', default='sgd', choices=['adam', 'sgd'], help="Optimizer")

    parser.add_argument('--samples', default=0, type=int, help="Number of training samples per class")
    parser.add_argument('--test_samples', default=0, type=int, help="Number of test samples per class")
    parser.add_argument('--dataset', default='CIFAR100', choices=['PMNIST', 'CIFAR100', 'CIFAR10', 'Imagenetr', 'CUB'], help="Name of the dataset to use")

    parser.add_argument('--reset', default=0, type=int, help="Whether to reset weights at the beginning of each tasks")
    parser.add_argument('--scenario', default='class', choices=['task', 'domain', 'class'],help="continual learning scenario")
    parser.add_argument('--tasks', default=10, type=int, help="Number of continual tasks")
    parser.add_argument('--classes', default=10, type=int, help="Number of classes per tasks")

    parser.add_argument('--mem', default=500, type=int, help="memory budget per class for data-replay. Set to zero for no memory")

    parser.add_argument('--fixed_mem', default=0, type=int, help="determines whether the memory size is fixed or grows with the number of tasks. If fixed, the total memory size is args.mem * tasks")
    parser.add_argument('--iid', default=0, type=int, help="whether to run the iid experiment")
    parser.add_argument('--exp_name', default='temp', type = str, help="Experiment name")

    parser.add_argument('--path', default='.', type=str, help="Root Path")
    parser.add_argument('--device', default='cuda', type=str, help="device name effective for neural network based models")

    parser.add_argument('--save', default=None, type=str, help="the unique name to save the results")
    parser.add_argument('--epochs', default=100, type=int, help="epochs per task")
    parser.add_argument('--seed', default=101, type=int, help="seed for reproducibility")

    parser.add_argument('--metrics', default=0, help="include more metrics such as forward and backward transfer")
    parser.add_argument('--overlap', default=0, help="determines if task samples overlap")
    parser.add_argument('--perm', default=0.0, type=float, help="permutation amount")
    parser.add_argument('--noise', default=0.0, type=float, help="noise amount")

    parser.add_argument('--save_back', default=None, type=str, help="save the backbone weights")
    parser.add_argument('--load_back', default=None, type=str, help="load the backbone weights")

    parser.add_argument('--save_mlp', default=None, type=str, help="save the mlp weights")

    parser.add_argument('--freeze_back', default=0, type=int, help="Freeze the backbone weights")

    if run_in_notebook:
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()



    return args

def train (model, train_dataloader):
    model.initialize_learning_()
    model.train(train_dataloader)
    res = model.finalize_learning_(train_dataloader)
    return res

def evaluate (model, test_dataloader):
    model.initialize_metrics_()
    for x,y in test_dataloader:
        y_pred, _ = model.classify(x, y)
    res = model.finalize_metrics_()

    return res



def main(args, verbose = False):
    from torch.utils.tensorboard import SummaryWriter
    if args.shared == [] and len(args.hidden) > 0:
        args.shared = [1] * len(args.hidden)

    print ("**************", args.device)

    # if args.device.startswith('cuda'):
    #     best_gpu = get_gpu_with_max_free_memory()
    #     args.device = f'cuda:{best_gpu}'
    
    set_seed(args.seed)

    writer = SummaryWriter(f"{args.path}/logs/{args.dataset}/{datetime.today().strftime('%Y%m%d%H%M%S')}_{args.exp_name}")
    cl_util = Continual_Util(args.classes, args.dataset, args.tasks, args.scenario)

    args.cl_util = cl_util

    if 'Imagenet' in args.dataset:
        args.num_input_channels = 3

    elif 'CUB' in args.dataset:
        args.num_input_channels = 3

    elif 'CIFAR' in args.dataset:
        args.num_input_channels = 3
    elif 'MNIST' in args.dataset:
        args.num_input_channels = 1
    else:
        raise NotImplementedError


    model = DNN(args)

    if args.freeze_back:
        args.embedding_function = model.net.backbone_
        cache_path = "embeddings/"
        os.makedirs(cache_path, exist_ok=True)

        pretrain_path = ""
        if args.load_back is not None:
            if os.path.exists(args.load_back):
                # get only file name
                pretrain_path = args.load_back.split("/")[-1].split(".")[0]
            else:
                pretrain_path = args.load_back


        args.embedding_cache_path = f"{cache_path}{args.dataset}_{args.backbone}_{args.k}_{pretrain_path}.torch"

    if args.dataset == 'Imagenetr':
        assert args.samples == 0, "Imagenet does not support samples count"
        assert args.test_samples == 0, "Imagenet does not support test samples count"

        dataset_train, dataset_test = get_Imagenetr(args, cl_util.NUM_CLASSES_PER_TASK, cl_util.NUM_TASKS, noise_ratio=args.noise)
    elif args.dataset == 'CUB':
        assert args.samples == 0, "Imagenet does not support samples count"
        assert args.test_samples == 0, "Imagenet does not support test samples count"

        dataset_train, dataset_test = get_CUB(args, cl_util.NUM_CLASSES_PER_TASK, cl_util.NUM_TASKS, noise_ratio=args.noise)

    elif args.dataset == 'PMNIST':
        dataset_train, dataset_test = get_PermutedMNIST(args, cl_util.NUM_CLASSES_PER_TASK, cl_util.NUM_TASKS, permute_ratio=args.perm, num_train=args.samples, num_test=args.test_samples, overlap_train_data = args.overlap, noise_ratio = args.noise)
    elif args.dataset == 'CIFAR100':
        dataset_train, dataset_test = get_CIFAR100(args, cl_util.NUM_CLASSES_PER_TASK, cl_util.NUM_TASKS, num_train=args.samples, num_test=args.test_samples, noise_ratio = args.noise)
    elif args.dataset == 'CIFAR10':
        dataset_train, dataset_test = get_CIFAR10(args, cl_util.NUM_CLASSES_PER_TASK, cl_util.NUM_TASKS, num_train=args.samples, num_test=args.test_samples, noise_ratio = args.noise)
    else:
        raise NotImplementedError

    args.workers = 0
    # if os.cpu_count() > 1 and os.name != 'nt':
    #     args.workers = min ([os.cpu_count(), 4])

    print("Using device", args.device, f'with {args.workers} workers.')


    model_name = model.name

    epoch = -1
    save_dict = {'test_loss':{},
                'test_acc':{},
                'train_loss':{},
                'train_acc':{},
                 'continual_accs':{},
                 'continual_losses':{},
                 'num_params': 0,
                 'task_epochs':[]
                 }

    for task_id in range (cl_util.NUM_TASKS):
        print ("========= Starting task", task_id)


        if args.iid :
            task_id =  cl_util.NUM_TASKS - 1

        cl_util.set_current_task(task_id)

        set_seed(args.seed + 1000 * task_id)

        train_classes = cl_util.get_classes_for_task(task_id) if not args.iid else cl_util.get_all_classes_before_task(cl_util.NUM_TASKS - 1)



        dataset_train.set_active_classes(train_classes)
        train_dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=args.bs, shuffle=True,
                                                       num_workers=args.workers, drop_last=False)

        dataset_test.set_active_classes(cl_util.get_all_classes_before_task(task_id))
        test_all_tasks_dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=args.bs, shuffle=False,
                                                                num_workers=args.workers, drop_last=False)

        model.initilaize_task_(task_id, train_dataloader, reset_weights= args.reset if not args.iid else 0)

        epochs = args.epochs
        # fixme
        if args.reset and task_id < args.tasks - 1:
            epochs = 1

        for i in tqdm(range(0,epochs)):

            set_seed(args.seed + 1000 * task_id + i)
            epoch += 1

            train_loss, train_accuracy = train(model, train_dataloader)

            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Accuracy/train', train_accuracy, epoch)
            save_dict['train_acc'][epoch]  = train_accuracy
            save_dict['train_loss'][epoch]  = train_loss


            test_loss, test_accuracy = evaluate(model, test_all_tasks_dataloader)

            torch.cuda.empty_cache()

            writer.add_scalar('Loss/test', test_loss, epoch)
            writer.add_scalar('Accuracy/test', test_accuracy, epoch)

            save_dict['test_acc'][epoch]  = test_accuracy
            save_dict['test_loss'][epoch]  = test_loss

            if verbose:
                print (f"task {task_id}, epoch {epoch}: {model_name} train loss, {train_loss}, test loss: {test_loss}, train_accuracy: {train_accuracy}, test accuracy: {test_accuracy}")

            writer.flush()

        model.finalize_task_(train_loader=train_dataloader)

        save_dict['task_epochs'].append(epoch)

        if args.metrics:
            save_dict['continual_accs'][task_id] = {}
            save_dict['continual_losses'][task_id] = {}

            for i in range(task_id + 1):
                dataset_test.set_active_classes(cl_util.get_classes_for_task(i))

                test_dataloader = torch.utils.data.DataLoader(dataset_test, batch_size=128, shuffle=False, num_workers=args.workers, drop_last=False)

                test_loss, test_accuracy = evaluate(model, test_dataloader)
                save_dict['continual_accs'][task_id][i] = test_accuracy
                save_dict['continual_losses'][task_id][i] = test_loss




        if args.iid:
            break

    if args.save is not None:
        save_dict['num_params'] = model.get_num_params()
        Path(f"{args.path}/save/").mkdir(parents=True, exist_ok=True)
        torch.save(save_dict, f"{args.path}/save/{args.save}.torch")

    model.save_weights()

    writer.close()



if __name__=="__main__":
    args = get_args()
    main(args, verbose = True)