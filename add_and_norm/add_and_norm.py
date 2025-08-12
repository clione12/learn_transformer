from layer_norm import LayerNorm
import torch
from torch import nn

class AddAndNorm(nn.Module):
    def __init__(self, embed_size, dropout=0.1):
        super(AddAndNorm,self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNorm(embed_size=embed_size)

    def forward(self, x: torch.Tensor, sublayer: nn.Module):
        """
            这里原论文是这么做的，实际上layer_norm 在哪里是个可以讨论的问题
            layer_norm(x + dropout(sublayer(x)))
        """
        self.layer_norm(
            x + self.dropout(sublayer(x)) 
        )