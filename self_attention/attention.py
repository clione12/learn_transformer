import torch
import torch.nn as nn
from .scaled_dot_product_attention import scaled_dot_product_attention


class Attention(nn.Module):
    def __init__(self,embed_size: int):
        super(Attention,self).__init__()
        self.embed_size = embed_size
        self.W_q = nn.Linear(embed_size,embed_size)
        self.W_k = nn.Linear(embed_size,embed_size)
        self.W_v = nn.Linear(embed_size,embed_size)
        
    def forward(
        self, q: torch.Tensor, k: torch.Tensor ,v: torch.Tensor , mask: torch.Tensor = None
    ):
        Q = self.W_q(q)
        K = self.W_q(q)
        V = self.W_q(q)
        out, scores = scaled_dot_product_attention(Q, K, V, mask)

        return out, scores
