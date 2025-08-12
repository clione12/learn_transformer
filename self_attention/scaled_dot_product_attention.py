import torch
from torch import Tensor
import torch.nn.functional as F
import math

def scaled_dot_product_attention(
        Q: Tensor,
        K: Tensor,
        V: Tensor,
        mask: Tensor | None = None
):
    """
        attention(Q,K,V) = softmax(Q matmul K.T / sqrt(d_k)) ^ V
        Q: (batch, s, d) or (batch, h, s, d)
        K: (batch, s, d) or (batch, h, s, d)
        V: (batch, s, d) or (batch, h, s, d)
        mask: (batch, s, d) or (batch, h, s, d)

        return:
        (score, attention)
    """

    embed_size = Q.size(-1)

    score = F.softmax(
        torch.matmul(
            Q, K.transpose(-2,-1)
        )/ math.sqrt(embed_size), dim=-1
    )

    if mask is not None:
        score = score.masked_fill(mask == 0, float('-inf'))

    attention = torch.matmul(score, V)

    return attention, score


def test_scaled_dot_product_attention():
    Q = torch.randn(64, 100, 768)
    K = torch.randn(64, 100, 768)
    V = torch.randn(64, 100, 768)

    attention, score = scaled_dot_product_attention(Q,K,V)
    print(f"score:{score.size()}, attention:{attention.size()}")


if __name__ == "__main__":
    test_scaled_dot_product_attention()