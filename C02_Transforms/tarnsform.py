from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("./logs")

image_url = "example.jpg"
img = Image.open(image_url)

# 转换为Tensor
tensor_trans = transforms.ToTensor()
img_tensor = tensor_trans(img)
writer.add_image("img_tensor", img_tensor)

print(img_tensor.shape)

# 归一化 可以将数值限定在特定部分范围内 就是（x-mean）/std  不同的通道
norm_trans = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_normed = norm_trans(img_tensor)
writer.add_image("img_normed", img_normed)

# Resize
resize_trans = transforms.Resize((512, 512))  # 填写一个数的时候就是按短边进行等比缩放且不改变比例
img_resize = resize_trans(img_tensor)
print(img_resize.shape)
writer.add_image("img_resize", img_resize)

# Compose 就是一堆操作放一起
copmose_trans = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])
img_copmose = copmose_trans(img)
print(img_copmose.shape)
writer.add_image("img_copmose", img_copmose)

# RandomCrop
random_trans = transforms.RandomCrop(256)
random_compose = transforms.Compose([random_trans, tensor_trans])
for i in range(10):
    img_random_rot = random_compose(img)
    # print(img_random_rot.shape)
    writer.add_image("img_random_rot", img_random_rot,i)

writer.close()
