import torch
from torchvision.transforms import transforms

class ViTBackbone(torch.nn.Module):
    def __init__(self, name) -> None:
        super().__init__() 
        import timm
        b = timm.create_model(name, pretrained=True, num_classes=0)
        b.out_dim = 768
        self.backbone = b
        
        # for CIFAR dataset, we need to apply another transform
        self.transforms = transforms.Compose([
            transforms.Resize(224),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        
    def forward(self, x):
        if x.shape[-1] != 224:
            # print("Resizing image to 224x224")
            x = self.transforms(x)
        # print(f"Input shape: {x.shape}")
        return self.backbone(x)
    
    
    