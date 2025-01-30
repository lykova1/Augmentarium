import os
import random
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from augmentations import PixelDisplacementTransform

class AlbDataset(torch.utils.data.Dataset):
    """
    Обычный датасет, применяющий Albumentations к каждому изображению.
    """
    def __init__(self, root, transform=None):
        self.samples = torchvision.datasets.ImageFolder(root).samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = np.array(img)
            aug = self.transform(image=img)
            img = aug["image"]  # [C,H,W]
        else:
            img = torch.from_numpy(np.array(img).transpose(2,0,1))
        return img, label

class SamplePairingDataset(torchvision.datasets.ImageFolder):
    """
    Датасет, который с некоторой вероятностью
    смешивает текущее изображение с другим случайным (SamplePairing).
    """
    def __init__(self, root, transform=None, samplepairing_p=0.5):
        super().__init__(root, transform=None)
        self.alb_transform = transform
        self.samplepairing_p = samplepairing_p

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = Image.open(path).convert("RGB")
        image = np.array(image)
        if self.alb_transform is not None:
            aug1 = self.alb_transform(image=image)
            img_t = aug1["image"]
        else:
            img_t = torch.from_numpy(image.transpose(2,0,1))

        if random.random() < self.samplepairing_p:
            idx2 = random.randint(0, len(self.samples)-1)
            path2, _ = self.samples[idx2]
            image2 = Image.open(path2).convert("RGB")
            image2 = np.array(image2)
            if self.alb_transform is not None:
                aug2 = self.alb_transform(image=image2)
                img_t2 = aug2["image"]
            else:
                img_t2 = torch.from_numpy(image2.transpose(2,0,1))
            img_t = (img_t + img_t2)/2.0

        return img_t, target

class TorchVisionDataset(torchvision.datasets.ImageFolder):
    """
    Датасет, применяющий torchvision-трансформации (AutoAugment / RandAugment).
    """
    def __init__(self, root, tv_transform=None):
        super().__init__(root, transform=None)
        self.tv_transform = tv_transform

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.tv_transform:
            img = self.tv_transform(img)
        else:
            img = torchvision.transforms.ToTensor()(img)
        return img, label

# GridMask-класс вы можете оставить внутри augmentations.py или там же

from augmentations import make_albumentations
from models import GridMask  # если у вас GridMask в models, или augmentations

def build_dataset_and_loader(
    train_dir, val_dir, test_dir,
    manual_transform=None,  # Albumentations
    tv_transform=None,      # TorchVision
    batch_size=32,
    advanced_aug="None",  # "SamplePairing", "GridMask", ...
    samplepairing_p=0.5,
    gridmask_params=None
):
    """
    Собирает train/val/test DataLoader в зависимости от режима.
    """
    from models import GridMask  # Или из augmentations

    if tv_transform is not None:
        # AutoAugment / RandAugment
        if advanced_aug == "SamplePairing":
            # Несовместимо (упрощаем)
            advanced_aug = "None"

        train_dataset = TorchVisionDataset(train_dir, tv_transform=tv_transform)

    else:
        # Albumentations
        if advanced_aug == "SamplePairing":
            train_dataset = SamplePairingDataset(
                root=train_dir,
                transform=manual_transform,
                samplepairing_p=samplepairing_p
            )
        else:
            train_dataset = AlbDataset(train_dir, transform=manual_transform)

    # GridMask как collate_fn (если Alb)
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

    # Вал. и тест
    val_tf = A.Compose([
        A.Resize(64,64),
        A.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5)),
        ToTensorV2()
    ])
    val_dataset  = AlbDataset(val_dir,  transform=val_tf)
    test_dataset = AlbDataset(test_dir, transform=val_tf)

    val_loader  = DataLoader(val_dataset,  batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
