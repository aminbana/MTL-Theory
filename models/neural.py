import os

import numpy as np
import torch

from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from continual_utils import Continual_Util
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from datasets.Dataset import ReplayBuffer
from models.architectures.LinearBlock import LinearBlock
from models.architectures.Network import Network
from models.architectures.Resnet import ResNet18


def get_optimizer(args, net):
    if args.optim == 'adam':
        return torch.optim.Adam(net.parameters(),lr=args.lr)
    elif args.optim == 'sgd':
        return torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.95, nesterov=True)



class DNN:
    def __init__(self, args):
        self.name = 'DNN'

        self.device = args.device
        self.args = args
        self.cl_util:Continual_Util = self.args.cl_util

        self.net = self.make_network(args)

        self.optimizer = get_optimizer(args, self.net)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.mem_budget = args.mem

        if self.mem_budget > 0:
            self.replay_buffer = ReplayBuffer(args.mem * args.cl_util.NUM_CLASSES_PER_TASK, args.tasks, args.fixed_mem)

    def make_network(self, args):
        model = Network(args)
        return model.to(self.device)

    def classify(self, x, y):
        y, task_ids = self.cl_util.modify_labels(y)
        with torch.no_grad():
            x = x.to(self.device)
            y = y.to(self.device)
            task_ids = task_ids.to(self.device)

            pred = self.net(x, task_ids)

            loss = self.criterion(pred, y)

            self.total_loss += loss.item() * len(x)
            self.count += len(x)

            y_pred = pred.argmax(-1)
            self.total_accuracy += (y_pred == y).sum().to('cpu').item()
        return y_pred.to('cpu'), pred.cpu().detach()

    def update_train_loader_with_mem(self, train_loader:DataLoader):
        if self.mem_budget > 0 and self.task_id > 0:
            new_dataset = torch.utils.data.ConcatDataset([train_loader.dataset, self.replay_buffer])
            new_dataloader = DataLoader(new_dataset, batch_size=train_loader.batch_size, shuffle=True, num_workers=train_loader.num_workers)
            return new_dataloader
        else:
            return train_loader


    def train(self, train_dataloader):

        train_dataloader = self.update_train_loader_with_mem(train_dataloader)

        for x, y in train_dataloader:
            y, task_ids = self.cl_util.modify_labels(y)
            x = x.to(self.device)
            y = y.to(self.device)
            task_ids = task_ids.to(self.device)
            self.optimizer.zero_grad()
            pred = self.net(x, task_ids)
            loss = self.criterion(pred, y)
            loss.backward()
            self.optimizer.step()

            self.total_loss += loss.item() * len(x)
            self.count += len(x)

            y_pred = pred.argmax(-1)
            self.total_accuracy += (y_pred == y).sum().to('cpu').item()

    def initilaize_task_(self, task_id, train_loader, reset_weights):
        self.task_id = task_id

        if reset_weights:
            self.net = self.make_network(self.args)
            self.optimizer = get_optimizer(self.args, self.net)

    def finalize_task_(self, train_loader):
        if self.mem_budget > 0:
            self.replay_buffer.sample_task(train_loader.dataset)

    def save_weights(self):
        self.net.save_back(self.args.save_back)
        self.net.save_mlp(self.args.save_mlp)

    def initialize_metrics_(self):
        self.total_loss = 0
        self.total_accuracy = 0
        self.count = 0

    def finalize_metrics_(self):
        total_loss = self.total_loss / self.count
        total_accuracy = self.total_accuracy / self.count
        return total_loss, total_accuracy

    def initialize_learning_(self):
        self.net.train()
        return self.initialize_metrics_()

    def finalize_learning_(self, train_loader):
        self.net.eval()
        return self.finalize_metrics_()

    def get_num_params(self):
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)
