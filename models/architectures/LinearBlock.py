import torch
from torch import nn
import torch.nn.functional as F

class LinearBlock(nn.Module):
    def __init__(self, input_dim, out_dim, batch_norm = True, activation = nn.ReLU, num_tasks = 1):
        super(LinearBlock, self).__init__()
        self.is_shared = (num_tasks == 1)
        self.num_tasks = num_tasks
        self.fc = nn.ModuleList([nn.Linear(input_dim, out_dim) for _ in range(num_tasks)])
        self.activation = nn.Identity() if activation is None else activation()
        self.batch_norm = nn.ModuleList([nn.BatchNorm1d(out_dim) if batch_norm else nn.Identity() for _ in range(num_tasks)])

    def forward(self, x, task_ids):
        if self.is_shared:
            x = self.fc[0](x)
            x = self.batch_norm[0](x)
        else:
            output = torch.zeros(x.shape[0], self.fc[0].out_features, device = x.device)
            for t in range(self.num_tasks):
                mask = task_ids == t
                if mask.sum() > 0:
                    output[mask] = self.fc[t](x[mask])
                    output[mask] = self.batch_norm[t](output[mask])
            x = output

        x = self.activation(x)
        return x


