import torch
import torch.nn as nn
from torch import Tensor

from scaled_dot_product_attention import scaled_dot_product_attention

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size: int, head_num: int):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.head_num = head_num

        self.Q_W = nn.Linear(embed_size, embed_size)
        self.K_W = nn.Linear(embed_size, embed_size)
        self.V_W = nn.Linear(embed_size, embed_size)
        self.fc = nn.Linear(embed_size, embed_size)

    

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None):
        """
            q: (batch_size, s, embed_size)
        """
        # Q: (batch_size, seq_size, head_num, embed_size)
        batch_size = q.size(0)
        seq_size = q.size(1)

        # q: (batch_size, num_head, seq_size, embed_size)
        Q = self.Q_W(q).view(batch_size, seq_size, self.head_num, -1).transpose(1,2)
        K = self.K_W(k).view(batch_size, seq_size, self.head_num, -1).transpose(1,2)
        V = self.V_W(v).view(batch_size, seq_size, self.head_num, -1).transpose(1,2)

        # out: (batch_size, num_head, seq_size, embed_size)
        scaled_attention, _ = self.scaled_dot_product_attention(Q, K, V, mask)
        concat_out = scaled_attention.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_size)

        out = self.fc(concat_out)

        return out


        