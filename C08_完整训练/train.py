# By: https://github.com/tyza66
import torch
import torchvision
import torch.nn as nn  # 或 from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 加载数据
train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=False,
                                         transform=torchvision.transforms.ToTensor())

train_data_size = len(train_data)
test_data_size = len(test_data)

train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 修改使用的设备 （有cuda用cuda）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)


# 模型定义
class CIFAR10Model(nn.Module):
    def __init__(self):
        super().__init__()
        # 各层序列
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


# 创建网络模型对象
model = CIFAR10Model().to(device)
# 损失函数
loss_fn = nn.CrossEntropyLoss()
# 优化器
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# tensorboard
writer = SummaryWriter("./logs")
# 训练网络参数
total_train_step = 0
total_test_step = 0
epochs = 10

# 训练
for epoch in range(epochs):
    print('训练轮数 Epoch {}/{}'.format(epoch + 1, epochs))

    model.train()
    for data in train_dataloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)

        # 优化器
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            writer.add_scalar('Loss/train', loss.item(), total_train_step)
            print('Loss/train 第 {} 轮训练损失率 {}'.format(total_train_step, loss.item()))

    model.eval()
    total_test_loss = 0
    total_test_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            images, labels = data  # images是图像 labels是每个图像对应的标签（从0开始的下标）
            images, targets = images.to(device), labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum() # 结果中是预测结果是每种情况的概率
            total_test_accuracy += accuracy
            total_test_step += 1
            if total_test_step % 100 == 0:
                writer.add_scalar('Loss/test', loss.item(), total_test_step)
                print('Loss/test 第 {} 轮测试损失率 {}'.format(total_test_step, loss.item()))

    print('整体测试集上的Loss: {}'.format(total_test_loss))
    print('整体测试集上的Accuracy: {}'.format(total_test_accuracy / test_data_size))
    writer.add_scalar("Total Loss/test", total_test_loss, epoch + 1)
    writer.add_scalar('Total Accuracy/test', total_test_accuracy / test_data_size, epoch + 1)

    torch.save(model,"model_{}.pth".format(epoch))
    #torch.save(model.state_dict(), 'checkpoint.pth')

# 关闭tensorboard的writer
writer.close()
