import torch
from torch import nn
from torch import Tensor

class LayerNorm(nn.Module):
    def __init__(self, embed_size: int, epsilon: float = 1e-9):
        super(LayerNorm, self).__init__()

        self.embed_size = embed_size
        self.epsilon = epsilon

        self.gamma = nn.Parameter(torch.ones(self.embed_size))

        self.beta = nn.Parameter(torch.zeros(self.embed_size))

    def forward(self, x: Tensor):
        """
        标准化流程
            x_std = (x - mean) / (std + epsilon)
            out = self.gamma * x_std  + self.beta
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)

        x_std = (x - mean) / (std + self.epsilon)
        return self.gamma * x_std  + self.beta
