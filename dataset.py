import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image
import copy
from tqdm import tqdm

class NoisyMNISTDatasetBinary(Dataset):
    def __init__(self, root_dir, train, t = None):
        self.transform = t
        if train:
            npz_file = os.path.join(root_dir, "train.npz")
        else:
            npz_file = os.path.join(root_dir, "test.npz")

        data = np.load(npz_file)

        self.noise = data["noise"]
        self.train = train
        if self.train:
            self.origin = data["origin"]

        
        # 如果需要数据增强
        if self.train:
            tsf_noise = copy.deepcopy(self.noise)
            tsf_origin = copy.deepcopy(self.origin)
            # 遍历每一张图片，添加进度条
            for i in tqdm(range(len(tsf_noise)), desc="Data Augmentation"):
                # 将numpy数组转换为PIL Image，以便transforms处理
                x = Image.fromarray(tsf_noise[i])
                y = Image.fromarray(tsf_origin[i])
                # 随机种子，确保噪声图和原图应用相同的随机变换
                seed = np.random.randint(2147483647)
                torch.manual_seed(seed)
                x = self.transform(x)
                torch.manual_seed(seed)
                y = self.transform(y)
                # 将增强后的数据重新赋值回数组
                tsf_noise[i] = np.array(x)
                tsf_origin[i] = np.array(y)
            # 合并数据集
            self.noise = np.concatenate((self.noise, tsf_noise), axis=0)
            self.origin = np.concatenate((self.origin, tsf_origin), axis=0)
        
    def __len__(self):
        return self.noise.shape[0]

    def __getitem__(self, idx):
        noise = torch.from_numpy(self.noise[idx]).float().unsqueeze(0) / 255.0
        if not self.train:
            return noise
        origin = torch.from_numpy(self.origin[idx]).float().unsqueeze(0) / 255.0
        return noise, origin

def dataloader(batch_size):
    # 训练集的数据增强
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomVerticalFlip(),    # 随机垂直翻转
        transforms.RandomRotation(45),      # 随机旋转，范围-45到+45度
        transforms.RandomResizedCrop(28, scale=(0.8, 1.0), ratio=(0.9, 1.1)),  # 随机裁切并缩放
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 随机调整亮度对比度，对灰度图可能效果不明显，但对有噪点的情况可能有帮助
    ])

    full_train_dataset = NoisyMNISTDatasetBinary("./dataset", True, t = train_transform)
    dataset_size = len(full_train_dataset)
    val_size = int(0.1 * dataset_size)
    train_size = dataset_size - val_size
    # 划分训练集和验证集
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    test_dataset = NoisyMNISTDatasetBinary("./dataset", False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader