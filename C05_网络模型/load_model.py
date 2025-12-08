import torchvision
from torch import nn
from torchvision.models import VGG16_Weights


vgg16_false = torchvision.models.vgg16(weights=None) # 不加载预训练模型
vgg16_true = torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1) # 加载预训练模型

print(vgg16_true)

train_data = torchvision.datasets.CIFAR10('./data', train=True, transform=torchvision.transforms.ToTensor(),
                                          download=True)

# 添加预训练模型的最后一层分类器
vgg16_true.classifier.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_true)

# 修改不加载预训练模型的最后一层分类器
print(vgg16_false)
vgg16_false.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false)

