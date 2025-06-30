import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms # 导入transforms模块
from PIL import Image

class NoisyMNISTDatasetBinary(Dataset):
    transform = None

    def __init__(self, root_dir, train): # 添加transform参数
        if train:
            npz_file = os.path.join(root_dir, "train.npz")
        else:
            npz_file = os.path.join(root_dir, "test.npz")

        if not os.path.exists(npz_file):
            print(f"Warning: {npz_file} not found. Creating dummy data.")
            if train:
                dummy_noise = np.random.randint(0, 256, (1000, 28, 28), dtype=np.uint8)
                dummy_origin = np.random.randint(0, 256, (1000, 28, 28), dtype=np.uint8)
                np.savez_compressed(npz_file, noise=dummy_noise, origin=dummy_origin)
            else:
                dummy_noise = np.random.randint(0, 256, (200, 28, 28), dtype=np.uint8)
                np.savez_compressed(npz_file, noise=dummy_noise)

        data = np.load(npz_file)
        self.noise = data["noise"]
        self.train = train 
        if self.train:
            self.origin = data["origin"]

    def __len__(self):
        return self.noise.shape[0]

    def __getitem__(self, idx):
        # 将numpy数组转换为PIL Image，以便transforms处理
        noise_img = Image.fromarray(self.noise[idx])
        
        if self.train:
            origin_img = Image.fromarray(self.origin[idx])
            
            if self.transform:
                # 随机种子，确保噪声图和原图应用相同的随机变换
                seed = np.random.randint(2147483647) 
                
                torch.manual_seed(seed)
                noise_img = self.transform(noise_img)
                
                torch.manual_seed(seed)
                origin_img = self.transform(origin_img)
            
            # 转换为Tensor并归一化
            noise = transforms.ToTensor()(noise_img) 
            origin = transforms.ToTensor()(origin_img)
            
            return noise, origin
        else:
            # 测试集无需原图，也无需数据增强
            noise = transforms.ToTensor()(noise_img)
            return noise

def dataloader(batch_size):
    # 训练集的数据增强
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(), # 随机水平翻转
        transforms.RandomVerticalFlip(),   # 随机垂直翻转
        transforms.RandomRotation(15),     # 随机旋转，范围-15到+15度
        # transforms.RandomResizedCrop(28, scale=(0.8, 1.0), ratio=(0.9, 1.1)), # 随机裁切并缩放
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # 随机调整亮度对比度，对灰度图可能效果不明显，但对有噪点的情况可能有帮助
        # transforms.ToTensor(), # ToTensor在这里被移到了__getitem__中，以便统一处理
    ])

    # 验证集和测试集通常不进行随机数据增强，只进行必要的转换（如ToTensor）
    val_test_transform = transforms.Compose([
        # transforms.ToTensor(),
    ])

    full_train_dataset = NoisyMNISTDatasetBinary("./dataset", train=True)
    dataset_size = len(full_train_dataset)
    val_size = int(0.02 * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size]
    )

    train_dataset.transform = train_transform

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
    test_dataset = NoisyMNISTDatasetBinary("./dataset", train=False, transform=val_test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader