# datasets.py
import os
import random
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


class AlbDataset(torch.utils.data.Dataset):
    """
    Датасет, применяющий Albumentations для трансформации изображений.
    """
    def __init__(self, root: str, transform: A.Compose = None):
        self.samples = torchvision.datasets.ImageFolder(root).samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img_np = np.array(img)
            aug = self.transform(image=img_np)
            img = aug["image"]
        else:
            img = torch.from_numpy(np.array(img).transpose(2, 0, 1))
        return img, label


class SamplePairingDataset(torchvision.datasets.ImageFolder):
    """
    Датасет с реализацией Sample Pairing: случайное смешивание изображений.
    """
    def __init__(self, root: str, transform: A.Compose = None, samplepairing_p: float = 0.5):
        super().__init__(root, transform=None)
        self.alb_transform = transform
        self.samplepairing_p = samplepairing_p

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        image = Image.open(path).convert("RGB")
        image = np.array(image)
        if self.alb_transform:
            aug = self.alb_transform(image=image)
            img_t = aug["image"]
        else:
            img_t = torch.from_numpy(image.transpose(2, 0, 1))
        if random.random() < self.samplepairing_p:
            idx2 = random.randint(0, len(self.samples) - 1)
            path2, _ = self.samples[idx2]
            image2 = Image.open(path2).convert("RGB")
            image2 = np.array(image2)
            if self.alb_transform:
                aug2 = self.alb_transform(image=image2)
                img_t2 = aug2["image"]
            else:
                img_t2 = torch.from_numpy(image2.transpose(2, 0, 1))
            img_t = (img_t + img_t2) / 2.0
        return img_t, target


class TorchVisionDataset(torchvision.datasets.ImageFolder):
    """
    Датасет, использующий трансформации torchvision (AutoAugment/RandAugment).
    """
    def __init__(self, root: str, tv_transform=None):
        super().__init__(root, transform=None)
        self.tv_transform = tv_transform

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.tv_transform:
            img = self.tv_transform(img)
        else:
            from torchvision.transforms import ToTensor
            img = ToTensor()(img)
        return img, label


def build_dataset_and_loader(
    train_dir: str, val_dir: str, test_dir: str,
    manual_transform=None, tv_transform=None,
    batch_size: int = 32,
    advanced_aug: str = "None",
    samplepairing_p: float = 0.5,
    gridmask_params: dict = None
):
    """
    Собирает DataLoader для train, val и test наборов.
    """
    from models import GridMask

    if tv_transform is not None:
        # Для AutoAugment / RandAugment
        if advanced_aug == "SamplePairing":
            advanced_aug = "None"
        train_dataset = TorchVisionDataset(train_dir, tv_transform=tv_transform)
    else:
        if advanced_aug == "SamplePairing":
            train_dataset = SamplePairingDataset(
                root=train_dir,
                transform=manual_transform,
                samplepairing_p=samplepairing_p
            )
        else:
            train_dataset = AlbDataset(train_dir, transform=manual_transform)

    if advanced_aug == "GridMask" and tv_transform is None and gridmask_params is not None:
        class GridMaskTransform:
            def __init__(self, params):
                self.gm = GridMask(**params)
            def __call__(self, img_t):
                return self.gm(img_t)
        def collate_fn(batch):
            gm = GridMaskTransform(gridmask_params)
            new_batch = []
            for (img_t, lbl) in batch:
                new_batch.append((gm(img_t), lbl))
            return torch.utils.data.dataloader.default_collate(new_batch)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_tf = A.Compose([
        A.Resize(64, 64),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])
    val_dataset = AlbDataset(val_dir, transform=val_tf)
    test_dataset = AlbDataset(test_dir, transform=val_tf)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
