import torch
import torchvision
import random
import numpy as np

class GridMask:
    """
    Пример реализации GridMask (если хотите хранить его тут).
    """
    def __init__(self, d_min=10, d_max=50, r=0.2, p=0.7):
        self.d_min = d_min
        self.d_max = d_max
        self.r = r
        self.p = p

    def __call__(self, img_tensor):
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

def get_model(num_classes=2):
    model = torchvision.models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = torch.nn.Linear(512, num_classes)
    return model
