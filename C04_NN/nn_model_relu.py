import torch
import torchvision
from torch import nn
from torch.nn import ReLU, Sigmoid
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

input = torch.tensor([[1, -0.5],
                      [-1, 3]])

input = torch.reshape(input, (-1, 1, 2, 2))
print(input.shape)

dataset = torchvision.datasets.CIFAR10("./data", train=False, download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=64)

# 作用是为了给神经网路中引入非线性因素（特质）
class ReLUModel(nn.Module):
    def __init__(self):
        super(ReLUModel, self).__init__()
        self.relu1 = ReLU()  # ReLU激活函数 正数保留 负数变0
        self.sigmoid1 = Sigmoid() # Sigmoid激活函数 输出0-1之间的数值 负数趋近于0 正数趋近于1

    def forward(self, input):
        output = self.sigmoid1(input)
        return output

model = ReLUModel()

writer = SummaryWriter("./logs")
step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images("input", imgs, global_step=step)
    output = model(imgs)
    writer.add_images("output", output, step)
    step += 1

writer.close()

