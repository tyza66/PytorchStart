from torch.utils.data import Dataset
from PIL import Image
import os


class MyData(Dataset):
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_names = os.listdir(self.path)

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img_item_path = os.path.join(self.path, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_names)

ant_datasets = MyData('dataset/train','ants')
bee_datasets = MyData('dataset/train','bees')

print(len(ant_datasets))

img, label = ant_datasets[0]
print(img,label)
img.show()

# 数据集可以相加
train_dataset = ant_datasets + bee_datasets
print(len(train_dataset))

label = train_dataset[123]
print(label)

label = train_dataset[124]
print(label)


