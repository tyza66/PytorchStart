import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 从官方加载数据集合
train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=False,
                                         transform=torchvision.transforms.ToTensor())

print(len(train_data))
print(len(test_data))

print(train_data[0])
print(train_data.classes)

print(test_data[0])
print(test_data.classes)
img,label = test_data[0]
print(img.shape)
print(label)
img = transforms.ToPILImage()(img)
img.show()
print("图里是",train_data.classes[label])

writer = SummaryWriter("./logs")

# DataLoader 就像是扑克牌抽排  # shuffle是是否打乱  # num_workers是进程数  # drop_last是是否舍去除不开的数据
train_dataloader = DataLoader(dataset=train_data, batch_size=64, shuffle=True,num_workers=0,drop_last=False)

step = 0
for data in train_dataloader:
    images, labels = data
    writer.add_images("images", images, step)
    print(images.shape)
    print(labels)
    step += 1


writer.close()



