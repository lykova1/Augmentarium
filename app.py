import streamlit as st
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# PyTorch, TorchVision
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import (
    AutoAugment, AutoAugmentPolicy,
    RandAugment,
    Compose as TorchCompose,
    ToTensor as TorchToTensor,
    Normalize as TorchNormalize,
    Resize as TorchResize
)

# Albumentations
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2

from tqdm import tqdm


###########################################
#      ONE-HOT, ВИЗУАЛИЗАЦИЯ
###########################################
def to_one_hot(labels, num_classes=2):
    return torch.nn.functional.one_hot(labels, num_classes).float()


def visualize_images(images, labels, classes=("Benign", "Malignant"), num_images=5):
    plt.figure(figsize=(15, 4))
    B = images.size(0)
    for i in range(min(num_images, B)):
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = (img * 0.5 + 0.5).clip(0, 1)  # денормализация

        plt.subplot(1, num_images, i + 1)
        plt.imshow(img)

        # Проверка soft или hard labels
        if len(labels.shape) == 2 and labels.shape[1] == 2:
            val0 = labels[i, 0].item()
            val1 = labels[i, 1].item()
            text = f"{classes[0]}: {val0:.2f}\n{classes[1]}: {val1:.2f}"
        else:
            cls_idx = int(labels[i].item())
            text = f"{classes[cls_idx]}"

        plt.title(text, fontsize=12)
        plt.axis("off")

    st.pyplot(plt.gcf())
    plt.close()


###########################################
#        GridMask
###########################################
class GridMask:
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


###########################################
#   Random Pixel Displacement (кастом)
###########################################
class PixelDisplacementTransform(ImageOnlyTransform):
    def __init__(self, stddev=0.7, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.stddev = stddev

    def apply(self, image, **params):
        rows, cols, ch = image.shape
        displacement_x = np.random.normal(0, self.stddev, (rows, cols))
        displacement_y = np.random.normal(0, self.stddev, (rows, cols))
        x, y = np.meshgrid(np.arange(cols), np.arange(rows))
        map_x = np.clip(x + displacement_x, 0, cols - 1).astype(np.float32)
        map_y = np.clip(y + displacement_y, 0, rows - 1).astype(np.float32)
        distorted = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return distorted


###########################################
#   MixUp, CutMix, RICAP
###########################################
def mixup_data(inputs, labels, alpha=0.2, device='cpu'):
    if alpha <= 0:
        return inputs, labels, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = inputs.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_inputs = lam * inputs + (1 - lam) * inputs[index, :]
    labels_a, labels_b = labels, labels[index]
    return mixed_inputs, (labels_a, labels_b), lam


def cutmix_data(inputs, labels, alpha=0.2, device='cpu'):
    if alpha <= 0:
        return inputs, labels, 1.0
    lam = np.random.beta(alpha, alpha)
    B, C, H, W = inputs.size()
    index = torch.randperm(B).to(device)
    inputs_shuf = inputs[index, :]
    labels_a, labels_b = labels, labels[index]

    rx = np.random.randint(W)
    ry = np.random.randint(H)
    cut_w = int(W * np.sqrt(1 - lam))
    cut_h = int(H * np.sqrt(1 - lam))

    x1 = np.clip(rx - cut_w // 2, 0, W)
    x2 = np.clip(rx + cut_w // 2, 0, W)
    y1 = np.clip(ry - cut_h // 2, 0, H)
    y2 = np.clip(ry + cut_h // 2, 0, H)

    inputs[:, :, y1:y2, x1:x2] = inputs_shuf[:, :, y1:y2, x1:x2]

    area = (y2 - y1) * (x2 - x1)
    lam = 1 - area / float(W * H)
    return inputs, (labels_a, labels_b), lam


def ricap_data(inputs, labels, alpha=0.3, device='cpu'):
    if alpha <= 0:
        return inputs, labels, None
    B, C, H, W = inputs.shape
    cx = np.random.randint(1, W)
    cy = np.random.randint(1, H)
    w_ = [cx, W - cx, cx, W - cx]
    h_ = [cy, cy, H - cy, H - cy]
    new_inputs = torch.zeros_like(inputs)
    new_labels = torch.zeros((B, 4), dtype=torch.long, device=device)
    weights = torch.zeros((B, 4), dtype=torch.float, device=device)
    for b in range(B):
        for k in range(4):
            idx = random.randint(0, B - 1)
            x_k = np.random.randint(0, W - w_[k] + 1)
            y_k = np.random.randint(0, H - h_[k] + 1)
            x1, x2 = x_k, x_k + w_[k]
            y1, y2 = y_k, y_k + h_[k]
            if k == 0:  # top-left
                new_inputs[b, :, 0:h_[k], 0:w_[k]] = inputs[idx, :, y1:y2, x1:x2]
            elif k == 1:  # top-right
                new_inputs[b, :, 0:h_[k], w_[0]:w_[0] + w_[k]] = inputs[idx, :, y1:y2, x1:x2]
            elif k == 2:  # bottom-left
                new_inputs[b, :, h_[0]:h_[0] + h_[k], 0:w_[k]] = inputs[idx, :, y1:y2, x1:x2]
            else:  # bottom-right
                new_inputs[b, :, h_[1]:h_[1] + h_[k], w_[2]:w_[2] + w_[k]] = inputs[idx, :, y1:y2, x1:x2]
            new_labels[b, k] = labels[idx]
            weights[b, k] = w_[k] * h_[k] / float(W * H)
    return new_inputs, new_labels, weights


def mixup_criterion(criterion, outputs, labels_tuple, lam):
    labels_a, labels_b = labels_tuple
    return lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)


def cutmix_criterion(criterion, outputs, labels_tuple, lam):
    labels_a, labels_b = labels_tuple
    return lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)


def ricap_criterion(criterion, outputs, label_parts, weights):
    B = outputs.size(0)
    loss = 0.0
    for k in range(4):
        l_k = criterion(outputs, label_parts[:, k])
        loss += (weights[:, k] * l_k).mean(dim=0)
    return loss


###########################################
#       Модель (ResNet18)
###########################################
def get_model(num_classes=2):
    model = torchvision.models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(512, num_classes)
    return model


###########################################
#   SamplePairingDataset
###########################################
class SamplePairingDataset(torchvision.datasets.ImageFolder):
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
            img_t = torch.from_numpy(image.transpose(2, 0, 1))
        if random.random() < self.samplepairing_p:
            idx2 = random.randint(0, len(self.samples) - 1)
            path2, _ = self.samples[idx2]
            image2 = Image.open(path2).convert("RGB")
            image2 = np.array(image2)
            if self.alb_transform is not None:
                aug2 = self.alb_transform(image=image2)
                img_t2 = aug2["image"]
            else:
                img_t2 = torch.from_numpy(image2.transpose(2, 0, 1))
            img_t = (img_t + img_t2) / 2.0
        return img_t, target


###########################################
#   AlbDataset (обычный)
###########################################
class AlbDataset(torch.utils.data.Dataset):
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
            img = torch.from_numpy(np.array(img).transpose(2, 0, 1))
        return img, label


###########################################
#   TorchVisionDataset (для Auto/Rand)
###########################################
class TorchVisionDataset(torchvision.datasets.ImageFolder):
    def __init__(self, root, tv_transform=None):
        super().__init__(root, transform=None)
        self.tv_transform = tv_transform

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.tv_transform is not None:
            img = self.tv_transform(img)
        else:
            img = TorchToTensor()(img)
        return img, label


###########################################
#  make_albumentations (ручные)
###########################################
def make_albumentations(
        use_pixeldisp=False, pixel_disp_std=0.7,
        use_flip=False, flip_prob=0.5,
        use_vflip=False, vflip_prob=0.5,
        use_brightness=False, brightness_limit=0.2,
        use_rotation=False, max_rotation=30.0,
        use_shift_scale_rot=False, shift_limit=0.0625, scale_limit=0.1, rotate_limit=45,
        use_gaussnoise=False, gauss_varlimit=30.0,
        use_gaussianblur=False, blur_limit=3,
        use_motionblur=False, mblur_limit=3,
        use_sharpen=False, sharpen_alpha=0.3,
):
    t_list = [A.Resize(64, 64)]
    if use_pixeldisp:
        t_list.append(PixelDisplacementTransform(stddev=pixel_disp_std, p=0.5))
    if use_flip:
        t_list.append(A.HorizontalFlip(p=flip_prob))
    if use_vflip:
        t_list.append(A.VerticalFlip(p=vflip_prob))
    if use_brightness:
        t_list.append(A.RandomBrightnessContrast(brightness_limit=brightness_limit, p=0.5))
    if use_rotation and max_rotation > 0:
        t_list.append(A.Rotate(limit=max_rotation, p=0.5))
    if use_shift_scale_rot:
        t_list.append(A.ShiftScaleRotate(
            shift_limit=shift_limit, scale_limit=scale_limit,
            rotate_limit=rotate_limit,
            border_mode=cv2.BORDER_REFLECT, p=0.5
        ))
    if use_gaussnoise:
        t_list.append(A.GaussNoise(var_limit=(10.0, gauss_varlimit), p=0.5))
    if use_gaussianblur:
        t_list.append(A.GaussianBlur(blur_limit=(3, blur_limit), p=0.5))
    if use_motionblur:
        t_list.append(A.MotionBlur(blur_limit=(3, mblur_limit), p=0.5))
    if use_sharpen:
        t_list.append(A.Sharpen(alpha=(0.1, sharpen_alpha), p=0.5))

    t_list.append(A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    t_list.append(ToTensorV2())
    return A.Compose(t_list)


###########################################
#           build_dataset_and_loader
###########################################
def build_dataset_and_loader(
        train_dir, val_dir, test_dir,
        manual_transform=None,  # Albumentations
        tv_transform=None,  # TorchVision
        batch_size=32,
        advanced_aug="None",
        samplepairing_p=0.5,
        gridmask_params=None
):
    # advanced_aug может быть: None, SamplePairing, GridMask, MixUp, CutMix, RICAP
    # Но MixUp/CutMix/RICAP - делаются в train loop
    # => Здесь интересуют только SamplePairing (Dataset) и GridMask (collate_fn).

    # Выбираем dataset
    use_samplepairing = (advanced_aug == "SamplePairing")

    if tv_transform is not None:
        # AutoAugment / RandAugment
        if use_samplepairing:
            # SamplePairing + TorchVision (редкий случай)
            # Но SamplePairingDataset требует Albumentations transform...
            # Или мы скажем, что совместить SamplePairing + AutoAugment не получится?
            st.warning(
                "SamplePairing + AutoAugment (или RandAugment) одновременно не поддерживается! Отключаем SamplePairing.")
            use_samplepairing = False  # отключим
        # => просто TorchVisionDataset
        train_dataset = TorchVisionDataset(train_dir, tv_transform=tv_transform)

    else:
        # Albumentations
        if use_samplepairing:
            train_dataset = SamplePairingDataset(
                root=train_dir,
                transform=manual_transform,
                samplepairing_p=samplepairing_p
            )
        else:
            train_dataset = AlbDataset(train_dir, transform=manual_transform)

    # Если advanced_aug=="GridMask", collate_fn (только если Alb)
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

    # Val/test
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


###########################################
#           TRAIN LOOP
###########################################
def train_model(
        train_loader, val_loader, device, epochs=5,
        advanced_aug="None",
        alpha_mixup=0.2, alpha_cutmix=0.2, alpha_ricap=0.3
):
    model = get_model(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for ep in range(epochs):
        model.train()
        run_loss = 0.0
        correct_train = 0
        total_train = 0

        for (inp, lbl) in train_loader:
            inp = inp.to(device)
            lbl = lbl.to(device)

            optimizer.zero_grad()

            if advanced_aug == "MixUp":
                lbl_oh = to_one_hot(lbl, num_classes=2).to(device)
                mixed_inp, (la, lb), lam_m = mixup_data(inp, lbl_oh, alpha=alpha_mixup, device=device)
                out = model(mixed_inp)
                loss = mixup_criterion(criterion, out, (la, lb), lam_m)

            elif advanced_aug == "CutMix":
                lbl_oh = to_one_hot(lbl, num_classes=2).to(device)
                cm_inp, (la, lb), lam_c = cutmix_data(inp, lbl_oh, alpha=alpha_cutmix, device=device)
                out = model(cm_inp)
                loss = cutmix_criterion(criterion, out, (la, lb), lam_c)

            elif advanced_aug == "RICAP":
                ric_inp, ric_lbl, ric_w = ricap_data(inp, lbl, alpha=alpha_ricap, device=device)
                out = model(ric_inp)
                if ric_w is not None:
                    loss = ricap_criterion(criterion, out, ric_lbl, ric_w)
                else:
                    loss = criterion(out, lbl)
            else:
                # None / GridMask / SamplePairing
                out = model(inp)
                loss = criterion(out, lbl)

            loss.backward()
            optimizer.step()
            run_loss += loss.item()

            _, pred = torch.max(out.data, 1)
            total_train += lbl.size(0)
            correct_train += (pred == lbl).sum().item()

        train_loss = run_loss / len(train_loader)
        train_acc = 100.0 * correct_train / total_train

        # Validation
        model.eval()
        val_run_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for (inp_val, lbl_val) in val_loader:
                inp_val, lbl_val = inp_val.to(device), lbl_val.to(device)
                out_val = model(inp_val)
                l_val = criterion(out_val, lbl_val)
                val_run_loss += l_val.item()

                _, pred_val = torch.max(out_val.data, 1)
                val_total += lbl_val.size(0)
                val_correct += (pred_val == lbl_val).sum().item()

        val_loss = val_run_loss / len(val_loader)
        val_acc = 100.0 * val_correct / val_total

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        st.write(f"Эпоха [{ep + 1}/{epochs}] "
                 f"- Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                 f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

    return model, (train_losses, val_losses, train_accs, val_accs)


###########################################
#           EVALUATE ON TEST
###########################################
def evaluate_on_test(model, test_loader, device):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for inp, lbl in test_loader:
            inp, lbl = inp.to(device), lbl.to(device)
            out = model(inp)
            _, pred = torch.max(out.data, 1)
            total += lbl.size(0)
            correct += (pred == lbl).sum().item()
    return 100.0 * correct / total


###########################################
#      Графики обучения
###########################################
def plot_curves(train_losses, val_losses, train_accs, val_accs):
    ep = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(ep, train_losses, label='Train Loss', marker='o')
    axes[0].plot(ep, val_losses, label='Val Loss', marker='o')
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(ep, train_accs, label='Train Acc', marker='o')
    axes[1].plot(ep, val_accs, label='Val Acc', marker='o')
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Acc (%)")
    axes[1].legend()
    axes[1].grid(True)

    st.pyplot(fig)
    plt.close()


###########################################
#              STREAMLIT APP
###########################################
def main():
    st.title("Streamlit: Manual Albumentations / AutoAugment / RandAugment")
    st.markdown("""
    **Общая логика**:
    1. Выберите режим аугментации: Manual (Albumentations), AutoAugment, RandAugment.
    2. При Manual появляется список чекбоксов.  
    3. В «Дополнительных аугментациях» выберите **одну** из [None, SamplePairing, GridMask, MixUp, CutMix, RICAP].
    4. Посмотрите пример, запустите обучение.
    """)

    dataset_root = st.text_input("Укажите путь к датасету (train/val/test):")
    if not dataset_root:
        st.warning("Сначала укажите путь к датасету.")
        return

    train_dir = os.path.join(dataset_root, "train")
    val_dir = os.path.join(dataset_root, "val")
    test_dir = os.path.join(dataset_root, "test")

    if (not os.path.exists(train_dir)) or (not os.path.exists(val_dir)) or (not os.path.exists(test_dir)):
        st.error("Не найдены папки train/ val/ test/ в указанной директории.")
        return

    # 1) Режим (None / AutoAugment / RandAugment)
    st.subheader("Режим аугментации:")
    aug_mode = st.radio("Выберите:", ["None (Manual)", "AutoAugment", "RandAugment"])

    # 2) Дополнительная аугментация (SamplePairing / GridMask / MixUp / CutMix / RICAP / None)
    st.subheader("Дополнительная аугментация:")
    advanced_aug = st.radio(
        "Одна из:",
        ["None", "SamplePairing", "GridMask", "MixUp", "CutMix", "RICAP"]
    )

    # Настройки GridMask, MixUp/CutMix/RICAP
    gridmask_params = {}
    alpha_mixup = 0.2
    alpha_cutmix = 0.2
    alpha_ricap = 0.3

    if advanced_aug == "GridMask":
        st.markdown("**Параметры GridMask**:")
        dmin = st.slider("d_min", 1, 50, 10)
        dmax = st.slider("d_max", 10, 100, 50)
        rr = st.slider("r (доля закрытия)", 0.0, 1.0, 0.2, 0.05)
        pp = st.slider("p (вероятность)", 0.0, 1.0, 0.7, 0.05)
        gridmask_params = dict(d_min=dmin, d_max=dmax, r=rr, p=pp)
    elif advanced_aug == "MixUp":
        alpha_mixup = st.slider("alpha MixUp", 0.0, 1.0, 0.2, 0.05)
    elif advanced_aug == "CutMix":
        alpha_cutmix = st.slider("alpha CutMix", 0.0, 1.0, 0.2, 0.05)
    elif advanced_aug == "RICAP":
        alpha_ricap = st.slider("alpha RICAP", 0.0, 1.0, 0.3, 0.05)

    # Если Manual, то показываем чекбоксы Albumentations
    manual_transform = None
    tv_transform = None

    if aug_mode == "None (Manual)":
        st.subheader("Albumentations (Manual) настройки:")

        use_pixeldisp = st.checkbox("Random Pixel Displacement")
        pixel_disp_std = 0.7
        if use_pixeldisp:
            pixel_disp_std = st.slider("stddev пикселей", 0.1, 5.0, 0.7, 0.1)

        flip_flag = st.checkbox("HorizontalFlip")
        flip_prob = 0.5
        if flip_flag:
            flip_prob = st.slider("Prob HorizontalFlip", 0.0, 1.0, 0.5, 0.05)

        vflip_flag = st.checkbox("VerticalFlip")
        vflip_prob = 0.5
        if vflip_flag:
            vflip_prob = st.slider("Prob VerticalFlip", 0.0, 1.0, 0.5, 0.05)

        brightness_flag = st.checkbox("RandomBrightnessContrast")
        brightness_limit = 0.2
        if brightness_flag:
            brightness_limit = st.slider("brightness_limit", 0.0, 1.0, 0.2, 0.05)

        rotation_flag = st.checkbox("RandomRotation")
        max_rotation = 30
        if rotation_flag:
            max_rotation = st.slider("Max rotation (deg)", 0.0, 90.0, 30.0, 5.0)

        ssr_flag = st.checkbox("ShiftScaleRotate")
        shift_limit = 0.0625
        scale_limit = 0.1
        rotate_limit = 45
        if ssr_flag:
            shift_limit = st.slider("Shift limit", 0.0, 0.3, 0.0625, 0.01)
            scale_limit = st.slider("Scale limit", 0.0, 0.5, 0.1, 0.01)
            rotate_limit = st.slider("Rotate limit", 0, 90, 45, 5)

        gauss_flag = st.checkbox("GaussNoise")
        gauss_varlimit = 30.0
        if gauss_flag:
            gauss_varlimit = st.slider("GaussNoise var_limit", 1.0, 100.0, 30.0, 1.0)

        gblur_flag = st.checkbox("GaussianBlur")
        gblur_limit = 3
        if gblur_flag:
            gblur_limit = st.slider("GaussianBlur limit", 3, 15, 3, 1)

        mblur_flag = st.checkbox("MotionBlur")
        mblur_limit = 3
        if mblur_flag:
            mblur_limit = st.slider("MotionBlur limit", 3, 15, 3, 1)

        sharpen_flag = st.checkbox("Sharpen")
        sharpen_alpha = 0.3
        if sharpen_flag:
            sharpen_alpha = st.slider("Sharpen alpha max", 0.1, 1.0, 0.3, 0.1)

        manual_transform = make_albumentations(
            use_pixeldisp=use_pixeldisp, pixel_disp_std=pixel_disp_std,
            use_flip=flip_flag, flip_prob=flip_prob,
            use_vflip=vflip_flag, vflip_prob=vflip_prob,
            use_brightness=brightness_flag, brightness_limit=brightness_limit,
            use_rotation=rotation_flag, max_rotation=max_rotation,
            use_shift_scale_rot=ssr_flag, shift_limit=shift_limit, scale_limit=scale_limit, rotate_limit=rotate_limit,
            use_gaussnoise=gauss_flag, gauss_varlimit=gauss_varlimit,
            use_gaussianblur=gblur_flag, blur_limit=gblur_limit,
            use_motionblur=mblur_flag, mblur_limit=mblur_limit,
            use_sharpen=sharpen_flag, sharpen_alpha=sharpen_alpha
        )
        tv_transform = None

    elif aug_mode == "AutoAugment":
        st.info("AutoAugment (torchvision)")
        policy_list = ["IMAGENET", "CIFAR10", "SVHN"]
        policy_pick = st.selectbox("Policy AutoAugment", policy_list)
        if policy_pick == "IMAGENET":
            policy = AutoAugmentPolicy.IMAGENET
        elif policy_pick == "CIFAR10":
            policy = AutoAugmentPolicy.CIFAR10
        else:
            policy = AutoAugmentPolicy.SVHN
        tv_transform = TorchCompose([
            TorchResize((64, 64)),
            AutoAugment(policy=policy),
            TorchToTensor(),
            TorchNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

    else:
        # RandAugment
        st.info("RandAugment (torchvision)")
        num_ops = st.slider("RandAugment num_ops", 1, 5, 2)
        magnitude = st.slider("RandAugment magnitude", 0, 10, 5)
        tv_transform = TorchCompose([
            TorchResize((64, 64)),
            RandAugment(num_ops=num_ops, magnitude=magnitude),
            TorchToTensor(),
            TorchNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
        manual_transform = None

    st.subheader("Показать пример аугментации")
    if st.button("Показать пример"):
        # Создаем DataLoader
        # advanced_aug может быть SamplePairing / GridMask / MixUp ...
        # Но MixUp/CutMix/RICAP логика в train loop, для отображения "смешанных" мы вручную вызываем.
        train_loader, val_loader, test_loader = build_dataset_and_loader(
            train_dir, val_dir, test_dir,
            manual_transform=manual_transform,
            tv_transform=tv_transform,
            batch_size=8,
            advanced_aug=advanced_aug,
            samplepairing_p=0.5,  # если SamplePairing, используется
            gridmask_params=gridmask_params if advanced_aug == "GridMask" else None
        )

        data_iter = iter(train_loader)
        images, labels = next(data_iter)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        images, labels = images.to(device), labels.to(device)

        classes = ["Benign", "Malignant"]

        # Если MixUp/CutMix/RICAP выбрано - покажем пример смешанных
        if advanced_aug == "MixUp":
            labels_oh = to_one_hot(labels, num_classes=2).to(device)
            mix_inp, (la, lb), lam_m = mixup_data(images, labels_oh, alpha=0.2, device=device)
            final_labels = lam_m * la + (1 - lam_m) * lb
            visualize_images(mix_inp, final_labels, classes=classes, num_images=5)
        elif advanced_aug == "CutMix":
            labels_oh = to_one_hot(labels, num_classes=2).to(device)
            cm_inp, (la, lb), lam_c = cutmix_data(images, labels_oh, alpha=0.2, device=device)
            cm_labels = lam_c * la + (1 - lam_c) * lb
            visualize_images(cm_inp, cm_labels, classes=classes, num_images=5)
        elif advanced_aug == "RICAP":
            ric_inp, ric_lbl, ric_w = ricap_data(images, labels, alpha=0.3, device=device)
            plt.figure(figsize=(15, 4))
            for i in range(5):
                img_np = ric_inp[i].cpu().numpy().transpose(1, 2, 0)
                img_np = (img_np * 0.5 + 0.5).clip(0, 1)
                plt.subplot(1, 5, i + 1)
                plt.imshow(img_np)
                seg_labels = ric_lbl[i]
                seg_weights = ric_w[i]
                txt = []
                for kk in range(4):
                    cls_idx = int(seg_labels[kk].item())
                    wv = seg_weights[kk].item()
                    txt.append(f"{classes[cls_idx]} ({wv:.2f})")
                plt.title("\n".join(txt), fontsize=10)
                plt.axis("off")
            st.pyplot(plt.gcf())
            plt.close()
        else:
            # None / SamplePairing / GridMask
            # просто визуализируем
            visualize_images(images, labels, classes=classes, num_images=5)

    st.subheader("Параметры обучения")
    ep = st.slider("Epochs", 1, 30, 5)
    bs = st.slider("Batch size", 4, 128, 32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.write("Используемое устройство:", device)

    if st.button("Начать обучение"):
        st.write("Создаём DataLoader...")
        train_loader, val_loader, test_loader = build_dataset_and_loader(
            train_dir, val_dir, test_dir,
            manual_transform=manual_transform,
            tv_transform=tv_transform,
            batch_size=bs,
            advanced_aug=advanced_aug,
            samplepairing_p=0.5,  # если SamplePairing
            gridmask_params=gridmask_params if advanced_aug == "GridMask" else None
        )
        st.write("Запуск train...")

        # Если выбрана MixUp/CutMix/RICAP -> передадим alpha
        alpha_mix = 0.2
        alpha_cut = 0.2
        alpha_ric = 0.3
        if advanced_aug == "MixUp":
            alpha_mix = 0.2
        elif advanced_aug == "CutMix":
            alpha_cut = 0.2
        elif advanced_aug == "RICAP":
            alpha_ric = 0.3

        model, stats = train_model(
            train_loader, val_loader, device, epochs=ep,
            advanced_aug=advanced_aug,
            alpha_mixup=alpha_mix,
            alpha_cutmix=alpha_cut,
            alpha_ricap=alpha_ric
        )
        train_losses, val_losses, train_accs, val_accs = stats

        st.subheader("Графики обучения")
        plot_curves(train_losses, val_losses, train_accs, val_accs)

        st.subheader("Точность на тесте")
        test_acc = evaluate_on_test(model, test_loader, device)
        st.write(f"Test Accuracy = {test_acc:.2f}%")

        st.success("Обучение завершено!")


if __name__ == "__main__":
    main()
