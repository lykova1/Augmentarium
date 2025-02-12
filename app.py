# app.py
import streamlit as st
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import logging

from augmentations import make_albumentations
from datasets import build_dataset_and_loader
from training import train_model, evaluate_on_test, mixup_data, cutmix_data, ricap_data
from models import get_model

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def to_one_hot(labels: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    """Преобразует метки классов в one-hot представление."""
    return torch.nn.functional.one_hot(labels, num_classes).float()


def plot_curves(train_losses, val_losses, train_accs, val_accs) -> None:
    """
    Отображает графики потерь и точности по эпохам.
    """
    ep = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # График Loss
    axes[0].plot(ep, train_losses, label='Train Loss', marker='o')
    axes[0].plot(ep, val_losses, label='Val Loss', marker='o')
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # График Accuracy
    axes[1].plot(ep, train_accs, label='Train Acc', marker='o')
    axes[1].plot(ep, val_accs, label='Val Acc', marker='o')
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True)

    st.pyplot(fig)
    plt.close()


def main():
    st.title("Augmentarium: Настройка аугментаций и обучение модели")
    st.markdown("""
    Добро пожаловать в **Augmentarium**!  
    Здесь вы можете:
    1. Загрузить датасет (он должен содержать папки train/, val/, test/).  
    2. Выбрать **ручные** аугментации (Albumentations) **или** автоматические (AutoAugment/RandAugment).  
    3. (Опционально) Включить «продвинутую» аугментацию (SamplePairing, GridMask, MixUp, CutMix, RICAP).  
    4. Посмотреть пример аугментированных изображений.  
    5. Запустить обучение и увидеть графики и финальную точность на тесте.  

    Просто настройте параметры, нажмите **"Показать пример"**, а затем **"Начать обучение"**.
    """)

    dataset_root = st.text_input("Укажите путь к датасету (train/ val/ test):")
    if not dataset_root:
        st.warning("Сначала укажите путь к датасету.")
        return

    train_dir = os.path.join(dataset_root, "train")
    val_dir = os.path.join(dataset_root, "val")
    test_dir = os.path.join(dataset_root, "test")

    if not (os.path.exists(train_dir) and os.path.exists(val_dir) and os.path.exists(test_dir)):
        st.error("Не найдены папки train/, val/, test/ в указанной директории.")
        return

    # --- Выбор режима аугментаций ---
    st.subheader("Режим (Manual Albumentations / AutoAugment / RandAugment)")
    aug_mode = st.radio("Выберите режим:", ["None (Manual)", "AutoAugment", "RandAugment"])

    advanced_aug = "None"
    gridmask_params = {}
    alpha_mixup = 0.2
    alpha_cutmix = 0.2
    alpha_ricap = 0.3

    if aug_mode == "None (Manual)":
        st.subheader("Доп. аугментация (SamplePairing/ GridMask/ MixUp/ CutMix/ RICAP)")
        advanced_aug = st.radio("Одну на выбор:", ["None", "SamplePairing", "GridMask", "MixUp", "CutMix", "RICAP"])
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
    else:
        st.info("Продвинутая аугментация отключена, т.к. выбран AutoAugment/RandAugment.")
        advanced_aug = "None"

    # --- Настройка трансформаций ---
    manual_transform = None
    tv_transform = None

    if aug_mode == "None (Manual)":
        st.subheader("Настройки базовых Albumentations:")
        use_flip = st.checkbox("HorizontalFlip")
        flip_prob = st.slider("Вероятность HorizontalFlip:", 0.0, 1.0, 0.5, 0.05) if use_flip else 0.5

        vflip_flag = st.checkbox("VerticalFlip")
        vflip_prob = st.slider("Вероятность VerticalFlip:", 0.0, 1.0, 0.5, 0.05) if vflip_flag else 0.5

        brightness_flag = st.checkbox("RandomBrightnessContrast")
        brightness_limit = st.slider("brightness_limit:", 0.0, 1.0, 0.2, 0.05) if brightness_flag else 0.2

        rotation_flag = st.checkbox("RandomRotation")
        max_rotation = st.slider("Макс угол поворота (градусы):", 0.0, 90.0, 30.0, 5.0) if rotation_flag else 30.0

        ssr_flag = st.checkbox("ShiftScaleRotate")
        if ssr_flag:
            shift_limit = st.slider("Shift limit:", 0.0, 0.3, 0.0625, 0.01)
            scale_limit = st.slider("Scale limit:", 0.0, 0.5, 0.1, 0.01)
            rotate_limit = st.slider("Rotate limit:", 0, 90, 45, 5)
        else:
            shift_limit, scale_limit, rotate_limit = 0.0625, 0.1, 45

        gauss_flag = st.checkbox("GaussNoise")
        gauss_varlimit = st.slider("GaussNoise var_limit:", 1.0, 100.0, 30.0, 1.0) if gauss_flag else 30.0

        gblur_flag = st.checkbox("GaussianBlur")
        gblur_limit = st.slider("GaussianBlur limit:", 3, 15, 3, 1) if gblur_flag else 3

        mblur_flag = st.checkbox("MotionBlur")
        mblur_limit = st.slider("MotionBlur limit:", 3, 15, 3, 1) if mblur_flag else 3

        sharpen_flag = st.checkbox("Sharpen")
        sharpen_alpha = st.slider("Sharpen alpha max:", 0.1, 1.0, 0.3, 0.1) if sharpen_flag else 0.3

        pixeldisp_flag = st.checkbox("Random Pixel Displacement")
        pixel_disp_std = st.slider("stddev для PixelDisplacement:", 0.1, 5.0, 0.7, 0.1) if pixeldisp_flag else 0.7

        manual_transform = make_albumentations(
            use_flip=use_flip, flip_prob=flip_prob,
            use_vflip=vflip_flag, vflip_prob=vflip_prob,
            use_brightness=brightness_flag, brightness_limit=brightness_limit,
            use_rotation=rotation_flag, max_rotation=max_rotation,
            use_shift_scale_rot=ssr_flag, shift_limit=shift_limit, scale_limit=scale_limit, rotate_limit=rotate_limit,
            use_gaussnoise=gauss_flag, gauss_varlimit=gauss_varlimit,
            use_gaussianblur=gblur_flag, blur_limit=gblur_limit,
            use_motionblur=mblur_flag, mblur_limit=mblur_limit,
            use_sharpen=sharpen_flag, sharpen_alpha=sharpen_alpha,
            use_pixeldisp=pixeldisp_flag, pixel_disp_std=pixel_disp_std
        )
        tv_transform = None

    elif aug_mode == "AutoAugment":
        st.subheader("Параметры AutoAugment (torchvision)")
        from torchvision.transforms import AutoAugment, AutoAugmentPolicy, Compose, Resize, ToTensor, Normalize
        policy_pick = st.selectbox("Policy:", ["IMAGENET", "CIFAR10", "SVHN"])
        policy = (AutoAugmentPolicy.IMAGENET if policy_pick == "IMAGENET"
                  else AutoAugmentPolicy.CIFAR10 if policy_pick == "CIFAR10"
                  else AutoAugmentPolicy.SVHN)
        tv_transform = Compose([
            Resize((64, 64)),
            AutoAugment(policy=policy),
            ToTensor(),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        manual_transform = None

    else:  # RandAugment
        st.subheader("Параметры RandAugment (torchvision)")
        from torchvision.transforms import RandAugment, Compose, Resize, ToTensor, Normalize
        num_ops = st.slider("RandAugment num_ops:", 1, 5, 2)
        magnitude = st.slider("RandAugment magnitude:", 0, 10, 5)
        tv_transform = Compose([
            Resize((64, 64)),
            RandAugment(num_ops=num_ops, magnitude=magnitude),
            ToTensor(),
            Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        manual_transform = None

    # --- Показ примера аугментации ---
    st.subheader("Показать пример аугментации")
    if st.button("Показать пример"):
        @st.cache_data(show_spinner=False)
        def load_data():
            return build_dataset_and_loader(
                train_dir, val_dir, test_dir,
                manual_transform=manual_transform,
                tv_transform=tv_transform,
                batch_size=8,
                advanced_aug=advanced_aug,
                gridmask_params=gridmask_params if advanced_aug == "GridMask" else None
            )
        train_loader, _, _ = load_data()

        data_iter = iter(train_loader)
        images, labels = next(data_iter)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        images, labels = images.to(device), labels.to(device)

        classes = ["Benign", "Malignant"]

        if advanced_aug == "MixUp":
            mix_inp, (la, lb), lam_m = mixup_data(images, to_one_hot(labels, 2).to(device), alpha_mixup, device=device)
            final_labels = lam_m * la + (1 - lam_m) * lb
            plt.figure(figsize=(15, 4))
            B = mix_inp.size(0)
            for i in range(min(5, B)):
                im = mix_inp[i].cpu().numpy().transpose(1, 2, 0)
                im = (im * 0.5 + 0.5).clip(0, 1)
                plt.subplot(1, 5, i + 1)
                plt.imshow(im)
                text = f"{classes[0]}: {final_labels[i, 0].item():.2f}\n{classes[1]}: {final_labels[i, 1].item():.2f}"
                plt.title(text)
                plt.axis("off")
            st.pyplot(plt.gcf())
            plt.close()

        elif advanced_aug == "CutMix":
            cm_inp, (la, lb), lam_c = cutmix_data(images, to_one_hot(labels, 2).to(device), alpha_cutmix, device=device)
            cm_labels = lam_c * la + (1 - lam_c) * lb
            plt.figure(figsize=(15, 4))
            B = cm_inp.size(0)
            for i in range(min(5, B)):
                im = cm_inp[i].cpu().numpy().transpose(1, 2, 0)
                im = (im * 0.5 + 0.5).clip(0, 1)
                plt.subplot(1, 5, i + 1)
                plt.imshow(im)
                text = f"{classes[0]}: {cm_labels[i, 0].item():.2f}\n{classes[1]}: {cm_labels[i, 1].item():.2f}"
                plt.title(text)
                plt.axis("off")
            st.pyplot(plt.gcf())
            plt.close()

        elif advanced_aug == "RICAP":
            ric_inp, ric_lbl, ric_w = ricap_data(images, labels, alpha_ricap, device=device)
            plt.figure(figsize=(15, 4))
            B = ric_inp.size(0)
            for i in range(min(5, B)):
                im = ric_inp[i].cpu().numpy().transpose(1, 2, 0)
                im = (im * 0.5 + 0.5).clip(0, 1)
                plt.subplot(1, 5, i + 1)
                plt.imshow(im)
                txt = []
                for kk in range(4):
                    cls_idx = ric_lbl[i][kk].item()
                    wv = ric_w[i][kk].item()
                    txt.append(f"{classes[int(cls_idx)]} ({wv:.2f})")
                plt.title("\n".join(txt))
                plt.axis("off")
            st.pyplot(plt.gcf())
            plt.close()
        else:
            plt.figure(figsize=(15, 4))
            B = images.size(0)
            for i in range(min(5, B)):
                im = images[i].cpu().numpy().transpose(1, 2, 0)
                im = (im * 0.5 + 0.5).clip(0, 1)
                plt.subplot(1, 5, i + 1)
                plt.imshow(im)
                plt.title(classes[int(labels[i].item())])
                plt.axis("off")
            st.pyplot(plt.gcf())
            plt.close()

    # --- Параметры обучения ---
    st.subheader("Параметры обучения")
    epochs = st.slider("Epochs", 1, 30, 5)
    batch_size = st.slider("Batch size", 4, 128, 32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.write("Используемое устройство:", device)

    if st.button("Начать обучение"):
        with st.spinner("Загрузка данных..."):
            train_loader, val_loader, test_loader = build_dataset_and_loader(
                train_dir, val_dir, test_dir,
                manual_transform=manual_transform,
                tv_transform=tv_transform,
                batch_size=batch_size,
                advanced_aug=advanced_aug,
                gridmask_params=gridmask_params if advanced_aug == "GridMask" else None
            )

        progress_bar = st.progress(0)
        status_text = st.empty()

        def progress_callback(epoch, total_epochs):
            progress_bar.progress((epoch + 1) / total_epochs)
            status_text.text(f"Эпоха {epoch + 1} из {total_epochs}")

        model, stats = train_model(
            train_loader, val_loader, device=device, epochs=epochs,
            advanced_aug=advanced_aug,
            alpha_mixup=alpha_mixup,
            alpha_cutmix=alpha_cutmix,
            alpha_ricap=alpha_ricap,
            progress_callback=progress_callback
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
