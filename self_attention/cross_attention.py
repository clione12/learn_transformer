import torch
import torch.nn as nn
from attention import Attention


class CrossAttention(nn.Module):
    def __init__(self, embed_size):
        super(CrossAttention,self).__init__()
        self.embed_size = embed_size
        self.attention = Attention(self.embed_size)

    def forward(
        self, 
        q: torch.Tensor,
        kv:torch.Tensor, 
        mask: torch.Tensor | None = None
    ):
        """
        参数
        q: (batch_size, s, embed_size)
        kv: (batch_size, s, embed_size)
        """
        out, score = self.attention(q,kv,kv,mask)

        return out, score
