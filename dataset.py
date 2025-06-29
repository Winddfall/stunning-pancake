import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


class NoisyMNISTDatasetBinary(Dataset):
    def __init__(self, root_dir, train): # 构造函数
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
        noise = torch.from_numpy(self.noise[idx]).float().unsqueeze(0) / 255.0
        if not self.train:
            return noise
        origin = torch.from_numpy(self.origin[idx]).float().unsqueeze(0) / 255.0
        return noise, origin


def dataloader(batch_size):
    full_train_dataset = NoisyMNISTDatasetBinary("./dataset", train=True) # 所有训练数据
    dataset_size = len(full_train_dataset)
    # val_size = int(0.1 * dataset_size) # 取10%用来验证
    val_size = int(0.02 * dataset_size) # 减少到2%用来验证
    train_size = dataset_size - val_size # 剩下的用来训练
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
    test_dataset = NoisyMNISTDatasetBinary("./dataset", train=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
