import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10("./data", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)

class MaxPoolModel(nn.Module):
    def __init__(self):
        super(MaxPoolModel, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=False) # kernel_size是池化窗口大小 并且默认步长是kernel_size ceil_mode是是否自动填充不足的边缘

    def forward(self, input):
        output = self.maxpool1(input)
        return output

model = MaxPoolModel()

writer = SummaryWriter("./logs")
step = 0

for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, step)
    output = model(imgs)
    writer.add_images("output", output, step)
    step = step + 1

writer.close()