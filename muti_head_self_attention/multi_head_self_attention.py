import torch.nn as nn
from torch import Tensor
from multi_head_attention import MultiHeadAttention

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_size: int, num_head: int):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.num_head = num_head
        self.muti_head_attention = MultiHeadAttention(
            embed_size, num_head
        )

    def forwad(self, x: Tensor, mask: Tensor = None):
        out = self.muti_head_attention(x,x,x,mask)
        return out
