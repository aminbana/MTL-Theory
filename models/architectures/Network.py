import os

import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights

from models.architectures.LinearBlock import LinearBlock
from models.architectures.Resnet import ResNet18
from models.architectures.ViT import ViTBackbone


class Network(nn.Module):
    def __init__(self, args):
        super(Network, self).__init__()

        if args.backbone == 'Resnet18':
            backbone = ResNet18(k=args.k, num_input_channels=args.num_input_channels)

            if args.load_back is not None:
                backbone.load_state_dict(torch.load(args.load_back))
                
            input_dim = 8 * args.k

        elif args.backbone == 'Resnet50':
            weights = args.load_back
            if weights is not None:
                if weights == 'IMAGENET1K_V2':
                    weights = ResNet50_Weights.IMAGENET1K_V2
                else:
                    assert False

            backbone = torchvision.models.resnet50(weights=args.load_back)
            backbone.fc = nn.Identity()
            
            input_dim = 2048

        elif args.backbone == 'ViTB16':
            import timm
            assert args.load_back in ["vit_base_patch16_224", "vit_base_patch16_224_in21k"], "Invalid backbone name"

            backbone = ViTBackbone(args.load_back)

            
            
            input_dim = 768

        else:
            raise NotImplementedError

        self.freeze_back = args.freeze_back
        self.backbone_ = None
        
        if args.freeze_back:
            backbone = backbone.eval()
            for param in backbone.parameters():
                param.requires_grad = False

            self.backbone_ = backbone
            self.backbone = nn.Identity()
        else:
            self.backbone = backbone

        
        layers_ = []
        for i in range(0, len(args.hidden)):
            layers_.append(LinearBlock(input_dim, args.hidden[i], batch_norm=True, activation = nn.ReLU, num_tasks= 1 if args.shared[i] == 1 else args.tasks))
            input_dim = args.hidden[i]


        self.fc = nn.Sequential(*layers_)


        if args.scenario == 'domain':
            num_heads = args.cl_util.NUM_CLASSES_PER_TASK
            shared_classifier = True
        elif args.scenario == 'task':
            num_heads = args.cl_util.NUM_CLASSES_PER_TASK
            shared_classifier = False
        elif args.scenario == 'class':
            num_heads = args.cl_util.NUM_TOTAL_CLASSES
            shared_classifier = True
        else:
            raise NotImplementedError

        self.classifier = LinearBlock(input_dim, num_heads, batch_norm = False, activation = None, num_tasks= 1 if shared_classifier else args.tasks)

        self.mlp = nn.ModuleList([*self.fc, self.classifier])

    def forward(self, x, task_ids):
        x = self.backbone(x)
        for layer in self.fc:
            x = layer(x, task_ids)
        x = self.classifier(x, task_ids)

        return x

    def save_back(self, path):
        if path is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(self.backbone.state_dict(), path)

    def save_mlp(self, path):
        if path is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(self.mlp.state_dict(), path)

    
    def eval(self):
        if self.backbone_ is not None:
            self.backbone_.eval()        
        self.backbone.eval()
        self.mlp.eval()
    
    
    def train(self):
        if self.backbone_ is not None and self.freeze_back:
            self.backbone_.eval()
        elif self.backbone_ is not None:
            self.backbone_.train()
            
        self.backbone.train()
        self.mlp.train()
    