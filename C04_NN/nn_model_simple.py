from torch import nn
import torch


class XXModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x + 1
        return x


model = XXModel()
x = torch.tensor(1.0)
y = model(x)
print(y)
