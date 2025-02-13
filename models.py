import random

import numpy as np
import torch
import torchvision


class GridMask:
    """
    Реализация GridMask для применения маски к изображению.
    """
    def __init__(self, d_min: int = 10, d_max: int = 50, r: float = 0.2, p: float = 0.7):
        self.d_min = d_min
        self.d_max = d_max
        self.r = r
        self.p = p

    def __call__(self, img_tensor: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img_tensor
        _, H, W = img_tensor.shape
        d = random.randint(self.d_min, self.d_max)
        l = int(d * self.r)
        mask = np.ones((H, W), dtype=np.float32)
        for i in range(0, H, d):
            for j in range(0, W, d):
                x_end = min(i + l, H)
                y_end = min(j + l, W)
                mask[i:x_end, j:y_end] = 0
        mask_t = torch.from_numpy(mask).to(img_tensor.device)
        return img_tensor * mask_t


def get_model(num_classes: int = 2) -> torch.nn.Module:
    """
    Возвращает модель ResNet18 с заменённым последним слоем для заданного числа классов.
    """
    model = torchvision.models.resnet18(pretrained=True)
    for param in model.parameters():
        param.require = False
    model.fc = torch.nn.Linear(512, num_classes)
    return model
