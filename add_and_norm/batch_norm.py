import torch
import torch.nn as nn
import torch.nn.functional as F

class BatchNorm1dManual(nn.Module):
    def __init__(self, num_features, epsilon=1e-5):
        super(BatchNorm1dManual, self).__init__()
        self.num_features = num_features
        self.epsilon = epsilon

        # 可学习的缩放参数 gamma 和 偏移参数 beta
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        # 计算批次中的均值和方差
        mean = x.mean(dim=0)
        var = x.var(dim=0, unbiased=False)

        # 执行归一化
        x_norm = (x - mean) / torch.sqrt(var + self.epsilon)

        # 缩放和平移
        out = self.gamma * x_norm + self.beta
        return out

if __name__ == "__main___":
    # 示例：批次大小为 3，特征维度为 4
    x = torch.randn(3, 4)  # 随机生成一个 3x4 的张量
    bn = BatchNorm1dManual(4)  # 创建一个具有 4 个特征的 BatchNorm 层
    output = bn(x)  # 前向传播
    print(output)