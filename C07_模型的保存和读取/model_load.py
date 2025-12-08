import torch
import torchvision

# 直接加载整个模型 就可以使用
model = torch.load("vgg16_method1.pth", weights_only=False)
print(model)

# 加载模型参数之后加载进模型结构 必须有原本模型的结构代码才能使用
vgg16 = torchvision.models.vgg16(weights=None)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))
print(vgg16)
