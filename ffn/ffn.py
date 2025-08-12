import torch
from torch import nn
from torch import Tensor

class PositionWizeFFN(nn.Module):

    def __init__(self, d_model: int, d_ff: int):
        super(PositionWizeFFN,self).__init__()
        self.W1 = nn.Linear(d_model,d_ff)
        self.W2 = nn.Linear(d_ff,d_model)
        self.relu = nn.functional.relu()

    def forward(self,x: Tensor):
        return self.W2(self.relu(self.W1(x)))
