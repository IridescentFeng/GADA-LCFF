from PIL import Image
import torch
from torch.utils.data import Dataset
import os

class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, root, txt, transform1=None):
        self.img_path = []
        self.labels = []
        self.transform1 = transform1

        with open(txt) as f:
            for line in f:
                self.img_path.append(root+line.split()[0].split("normal")[1])
                self.labels.append(int(line.split()[1]))
        self.targets = self.labels  # Sampler needs to use targets

        # self.images_path = images_path
        # self.images_class = images_class
        # self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        path = self.img_path[index]
        label = self.labels[index]
        # label = torch.tensor(label)
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
            sample1 = sample
            if sample.size[0]<=sample.size[1]:
                sample2 = sample.crop((0, 0, sample.size[0], sample.size[1] // 2))
                sample3 = sample.crop((0, sample.size[1] // 2, sample.size[0], sample.size[1]))
            else:
                sample2 = sample.crop((0, 0, sample.size[0]// 2, sample.size[1] ))
                sample3 = sample.crop((sample.size[0]//2, 0, sample.size[0], sample.size[1]))

        if self.transform1 is not None:
            sample1 = self.transform1(sample1)#0.5
            sample2 = self.transform1(sample2)
            sample3 = self.transform1(sample3)


        return sample1, sample2, sample3, label


    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
