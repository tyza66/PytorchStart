from torch import nn
import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())
train_dataloader = DataLoader(dataset=train_data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)


class Conv2dModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1,
                               padding=0)  # stride是步进大小 padding是边缘填充-默认0

    def forward(self, x):
        x = self.conv1(x)
        return x


model = Conv2dModel()

writer = SummaryWriter("./logs")

step = 0
for data in train_dataloader:
    imgs, targets = data
    outputs = model(imgs)
    print(imgs.shape)
    print(outputs.shape)

    outputs = torch.reshape(outputs, (-1, 3, 30, 30))
    writer.add_images("output", outputs, step)
    step = step + 1

writer.close()
