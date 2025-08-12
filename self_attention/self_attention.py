import torch
import torch.nn as nn
from attention import Attention


class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention,self).__init__()
        self.embed_size = embed_size
        self.attention = Attention(self.embed_size)

    def forward(self, x:torch.Tensor, mask: torch.Tensor | None = None):
        """
        参数
        x: (batch_size, s, embed_size)
        """
        out, score = self.attention(x,x,x,mask)

        return out, score
