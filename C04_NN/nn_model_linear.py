import torch
import torchvision
from torch import nn
from torch.nn import Linear
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10("./data", train=False, transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=64, drop_last=True)

class LinearMode(nn.Module):
    def __init__(self):
        super(LinearMode, self).__init__()
        self.linear1 = Linear(196608, 10) # 输入特征数 输出特征数 自动映射 196608=3*32*32*64
        # 默认是开启偏置的 bias=True

    def forward(self, input):
        output = self.linear1(input)
        return output

model = LinearMode()

for data in dataloader:
    imgs, targets = data
    print(imgs.shape)
    output = torch.flatten(imgs)
    print(output.shape)
    output = model(output)
    print(output.shape)