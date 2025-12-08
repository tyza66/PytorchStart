import torch
import torchvision
from torchvision.models import VGG16_Weights

vgg16 = torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
# 保存方模型结构+模型参数
torch.save(vgg16, "vgg16_method1.pth")

# 只保存模型参数（官方推荐） 只保存权重了 换环境的时候模型结构需要重新创建
torch.save(vgg16.state_dict(), "vgg16_method2.pth")
