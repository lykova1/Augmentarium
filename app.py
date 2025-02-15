import streamlit as st
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import logging

import optuna  # Для оптимизации

from visualization import show_augmented_images, plot_training_results

from augmentations import make_albumentations
from datasets import build_dataset_and_loader
from training import train_model, evaluate_on_test, mixup_data, cutmix_data, ricap_data
from models import get_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def to_one_hot(labels: torch.Tensor, num_classes: int = 2) -> torch.Tensor:
    """Преобразует метки классов в one-hot."""
    return torch.nn.functional.one_hot(labels, num_classes).float()


def main():
    st.title("Augmentarium: Настройка аугментаций и обучение модели")
    st.markdown("""
    Добро пожаловать в **Augmentarium**!  
    1. Загрузите датасет (train/, val/, test/).  
    2. Выберите аугментации (Albumentations) или AutoAugment/RandAugment.  
    3. Включите «продвинутую» аугментацию (SamplePairing, GridMask, MixUp, CutMix, RICAP) при необходимости.  
    4. Посмотрите аугментированные изображения.  
    5. Запустите обучение и смотрите результаты!
    """)

    # Храним «best_params» и «best_acc» в session_state
    if "best_params" not in st.session_state:
        st.session_state["best_params"] = {}
    if "best_acc" not in st.session_state:
        st.session_state["best_acc"] = None

    dataset_root = st.text_input("Укажите путь к датасету (train/ val/ test):")
    if not dataset_root:
        st.warning("Сначала укажите путь к датасету.")
        return

    train_dir = os.path.join(dataset_root, "train")
    val_dir   = os.path.join(dataset_root, "val")
    test_dir  = os.path.join(dataset_root, "test")

    if not (os.path.exists(train_dir) and os.path.exists(val_dir) and os.path.exists(test_dir)):
        st.error("Не найдены train/, val/, test/ в директории.")
        return

    st.subheader("Режим аугментации")
    aug_mode = st.radio("Режим:", ["None (Manual)", "AutoAugment", "RandAugment"])

    # Параметры продвинутой аугментации
    st.session_state.setdefault("advanced_aug", "None")
    st.session_state.setdefault("alpha_mixup", 0.2)
    st.session_state.setdefault("alpha_cutmix", 0.2)
    st.session_state.setdefault("alpha_ricap", 0.3)
    st.session_state.setdefault("grid_dmin", 10)
    st.session_state.setdefault("grid_dmax", 50)
    st.session_state.setdefault("grid_r", 0.2)
    st.session_state.setdefault("grid_p", 0.7)
    st.session_state.setdefault("samplepairing_p", 0.5)

    advanced_aug   = "None"
    gridmask_params= {}
    alpha_mixup    = 0.2
    alpha_cutmix   = 0.2
    alpha_ricap    = 0.3

    if aug_mode == "None (Manual)":
        st.subheader("Доп. аугментация")
        advanced_aug = st.radio("Одна из:", ["None", "SamplePairing", "GridMask", "MixUp", "CutMix", "RICAP"])
        st.session_state["advanced_aug"] = advanced_aug

        if advanced_aug == "GridMask":
            dmin_def = st.session_state["best_params"].get("gridmask_dmin", 10)
            dmax_def = st.session_state["best_params"].get("gridmask_dmax", 50)
            r_def    = st.session_state["best_params"].get("gridmask_r", 0.2)
            p_def    = st.session_state["best_params"].get("gridmask_p", 0.7)

            dmin = st.slider("d_min", 1, 50, dmin_def)
            dmax = st.slider("d_max", 10, 100, dmax_def)
            rr   = st.slider("r (доля закрытия)", 0.0, 1.0, float(r_def), 0.05)
            pp   = st.slider("p (вероятность)",   0.0, 1.0, float(p_def), 0.05)
            gridmask_params = dict(d_min=dmin, d_max=dmax, r=rr, p=pp)

        elif advanced_aug == "MixUp":
            alpha_mixup_def = st.session_state["best_params"].get("alpha_mixup", 0.2)
            alpha_mixup     = st.slider("alpha MixUp", 0.0, 1.0, float(alpha_mixup_def), 0.05)

        elif advanced_aug == "CutMix":
            alpha_cutmix_def = st.session_state["best_params"].get("alpha_cutmix", 0.2)
            alpha_cutmix     = st.slider("alpha CutMix", 0.0, 1.0, float(alpha_cutmix_def), 0.05)

        elif advanced_aug == "RICAP":
            alpha_ricap_def = st.session_state["best_params"].get("alpha_ricap", 0.3)
            alpha_ricap     = st.slider("alpha RICAP", 0.0, 1.0, float(alpha_ricap_def), 0.05)

        elif advanced_aug == "SamplePairing":
            sp_def = st.session_state["best_params"].get("samplepairing_p", 0.5)
            sp_val = st.slider("SamplePairing p", 0.0, 1.0, float(sp_def), 0.05)
            st.session_state["samplepairing_p"] = sp_val

    else:
        st.info("Продвинутая аугментация отключена (Auto/Rand)")
        advanced_aug = "None"

    # --- Настройка трансформаций ---
    manual_transform = None
    tv_transform     = None

    # Инициализируем флаги, если не в session_state
    if "use_flip" not in st.session_state:
        st.session_state["use_flip"]       = False
        st.session_state["use_vflip"]      = False
        st.session_state["use_brightness"] = False
        st.session_state["use_rotation"]   = False
        st.session_state["use_ssr"]        = False
        st.session_state["use_gaussnoise"] = False
        st.session_state["use_gblur"]      = False
        st.session_state["use_mblur"]      = False
        st.session_state["use_sharpen"]    = False
        st.session_state["use_pixeldisp"]  = False

    if aug_mode == "None (Manual)":
        st.subheader("Настройки Albumentations:")
        st.session_state["use_flip"]       = st.checkbox("HorizontalFlip",        value=st.session_state["use_flip"])
        st.session_state["use_vflip"]      = st.checkbox("VerticalFlip",          value=st.session_state["use_vflip"])
        st.session_state["use_brightness"] = st.checkbox("RandomBrightnessContrast", value=st.session_state["use_brightness"])
        st.session_state["use_rotation"]   = st.checkbox("RandomRotation",        value=st.session_state["use_rotation"])
        st.session_state["use_ssr"]        = st.checkbox("ShiftScaleRotate",      value=st.session_state["use_ssr"])
        st.session_state["use_gaussnoise"] = st.checkbox("GaussNoise",            value=st.session_state["use_gaussnoise"])
        st.session_state["use_gblur"]      = st.checkbox("GaussianBlur",          value=st.session_state["use_gblur"])
        st.session_state["use_mblur"]      = st.checkbox("MotionBlur",            value=st.session_state["use_mblur"])
        st.session_state["use_sharpen"]    = st.checkbox("Sharpen",               value=st.session_state["use_sharpen"])
        st.session_state["use_pixeldisp"]  = st.checkbox("Random Pixel Displacement", value=st.session_state["use_pixeldisp"])

        bestp = st.session_state["best_params"]
        flip_def      = bestp.get("flip_prob",        0.5)
        vflip_def     = bestp.get("vflip_prob",       0.5)
        bright_def    = bestp.get("brightness_limit", 0.2)
        rot_def       = bestp.get("rotation_limit",   30.0)
        shift_def     = bestp.get("shift_limit",      0.0625)
        scale_def     = bestp.get("scale_limit",      0.1)
        ssr_rot_def   = bestp.get("rotate_limit",     45)
        gauss_def     = bestp.get("gauss_varlimit",   30)
        gblur_def     = bestp.get("gblur_limit",      3)
        mblur_def     = bestp.get("mblur_limit",      3)
        sharpen_def   = bestp.get("sharpen_alpha",    0.3)
        pixel_def     = bestp.get("pixel_disp_std",   0.7)

        # Слайдеры
        flip_prob = flip_def
        if st.session_state["use_flip"]:
            flip_prob = st.slider("Вероятность HorizontalFlip:", 0.0, 1.0, float(flip_def), 0.05)

        vflip_prob = vflip_def
        if st.session_state["use_vflip"]:
            vflip_prob = st.slider("Вероятность VerticalFlip:", 0.0, 1.0, float(vflip_def), 0.05)

        brightness_limit = bright_def
        if st.session_state["use_brightness"]:
            brightness_limit = st.slider("brightness_limit:", 0.0, 1.0, float(bright_def), 0.05)

        max_rotation = rot_def
        if st.session_state["use_rotation"]:
            max_rotation = st.slider("Макс угол (градусы):", 0.0, 90.0, float(rot_def), 5.0)

        shift_limit  = shift_def
        scale_limit  = scale_def
        rotate_limit = ssr_rot_def
        if st.session_state["use_ssr"]:
            shift_limit = st.slider("Shift limit:", 0.0, 0.3, float(shift_def), 0.01)
            scale_limit = st.slider("Scale limit:", 0.0, 0.5, float(scale_def), 0.01)
            rotate_limit= st.slider("Rotate limit:", 0, 90,   int(ssr_rot_def), 5)

        gauss_varlimit = gauss_def
        if st.session_state["use_gaussnoise"]:
            gauss_varlimit = st.slider("GaussNoise var_limit:", 1.0, 100.0, float(gauss_def), 1.0)

        gblur_limit = gblur_def
        if st.session_state["use_gblur"]:
            gblur_limit = st.slider("GaussianBlur limit:", 3, 15, int(gblur_def), 1)

        mblur_limit = mblur_def
        if st.session_state["use_mblur"]:
            mblur_limit = st.slider("MotionBlur limit:", 3, 15, int(mblur_def), 1)

        sharpen_alpha = sharpen_def
        if st.session_state["use_sharpen"]:
            sharpen_alpha = st.slider("Sharpen alpha max:", 0.1, 1.0, float(sharpen_def), 0.1)

        pixel_disp_std = pixel_def
        if st.session_state["use_pixeldisp"]:
            pixel_disp_std = st.slider("stddev PixelDisplacement:", 0.1, 5.0, float(pixel_def), 0.1)

        from augmentations import make_albumentations
        manual_transform = make_albumentations(
            use_flip=st.session_state["use_flip"], flip_prob=flip_prob,
            use_vflip=st.session_state["use_vflip"], vflip_prob=vflip_prob,
            use_brightness=st.session_state["use_brightness"], brightness_limit=brightness_limit,
            use_rotation=st.session_state["use_rotation"], max_rotation=max_rotation,
            use_shift_scale_rot=st.session_state["use_ssr"],
            shift_limit=shift_limit, scale_limit=scale_limit, rotate_limit=rotate_limit,
            use_gaussnoise=st.session_state["use_gaussnoise"], gauss_varlimit=gauss_varlimit,
            use_gaussianblur=st.session_state["use_gblur"], blur_limit=gblur_limit,
            use_motionblur=st.session_state["use_mblur"], mblur_limit=mblur_limit,
            use_sharpen=st.session_state["use_sharpen"], sharpen_alpha=sharpen_alpha,
            use_pixeldisp=st.session_state["use_pixeldisp"], pixel_disp_std=pixel_disp_std
        )

        tv_transform = None

        # --- Автоподбор (Optuna) ---
        st.subheader("Автоматический подбор гиперпараметров (Optuna)")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Оптимизировать гиперпараметры"):
                with st.spinner("Оптимизация..."):
                    best_params, best_acc = optimize_aug_params(
                        train_dir, val_dir, test_dir,
                        use_flip=st.session_state["use_flip"],
                        use_vflip=st.session_state["use_vflip"],
                        use_brightness=st.session_state["use_brightness"],
                        use_rotation=st.session_state["use_rotation"],
                        use_ssr=st.session_state["use_ssr"],
                        use_gaussnoise=st.session_state["use_gaussnoise"],
                        use_gblur=st.session_state["use_gblur"],
                        use_mblur=st.session_state["use_mblur"],
                        use_sharpen=st.session_state["use_sharpen"],
                        use_pixeldisp=st.session_state["use_pixeldisp"],
                        advanced_aug=st.session_state["advanced_aug"]
                    )
                    st.session_state["best_params"] = best_params
                    st.session_state["best_acc"]    = best_acc
                st.success(f"Оптимизация завершена! val_acc={best_acc:.2f}, {best_params}")

        with col2:
            if st.button("Применить лучшие параметры"):
                if st.session_state["best_params"]:
                    st.success("Параметры применены! Обновите слайдеры.")
                else:
                    st.warning("Сначала запустите оптимизацию!")

    elif aug_mode == "AutoAugment":
        st.subheader("Параметры AutoAugment")
        from torchvision.transforms import AutoAugment, AutoAugmentPolicy, Compose, Resize, ToTensor, Normalize
        policy_pick = st.selectbox("Policy:", ["IMAGENET", "CIFAR10", "SVHN"])
        policy = AutoAugmentPolicy.IMAGENET
        if policy_pick == "CIFAR10":
            policy = AutoAugmentPolicy.CIFAR10
        elif policy_pick == "SVHN":
            policy = AutoAugmentPolicy.SVHN

        from torchvision.transforms import Compose
        tv_transform = Compose([
            Resize((64, 64)),
            AutoAugment(policy=policy),
            ToTensor(),
            Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
        ])
        manual_transform = None

    else: # RandAugment
        st.subheader("Параметры RandAugment")
        from torchvision.transforms import RandAugment, Compose, Resize, ToTensor, Normalize
        num_ops   = st.slider("RandAugment num_ops:", 1, 5, 2)
        magnitude = st.slider("RandAugment magnitude:",0, 10, 5)
        tv_transform = Compose([
            Resize((64, 64)),
            RandAugment(num_ops=num_ops, magnitude=magnitude),
            ToTensor(),
            Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
        ])
        manual_transform = None

    st.subheader("Показать пример аугментации")
    if st.button("Показать пример"):
        # Используем build_dataset_and_loader, который вернёт DataLoader с кастомным collate_fn,
        # если advanced_aug == "GridMask"
        train_loader, _, _ = build_dataset_and_loader(
                train_dir, val_dir, test_dir,
                manual_transform=manual_transform,
                tv_transform=tv_transform,
                batch_size=8,
                advanced_aug=st.session_state["advanced_aug"],
                gridmask_params=gridmask_params if st.session_state["advanced_aug"] == "GridMask" else None,
                samplepairing_p=st.session_state.get("samplepairing_p", 0.5)
            )

        adv = st.session_state["advanced_aug"]
        if adv in ["MixUp", "CutMix", "RICAP"]:
            data_iter = iter(train_loader)
            images, labels = next(data_iter)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            images, labels = images.to(device), labels.to(device)
            classes = ["Benign", "Malignant"]

            if adv == "MixUp":
                mix_inp, (la, lb), lam_m = mixup_data(images, to_one_hot(labels, 2).to(device), alpha_mixup, device=device)
                final_labels = lam_m * la + (1 - lam_m) * lb
                fig = plt.figure(figsize=(15, 4))
                for i in range(min(5, mix_inp.size(0))):
                    im = mix_inp[i].cpu().numpy().transpose(1, 2, 0)
                    im = (im*0.5 + 0.5).clip(0,1)
                    ax = fig.add_subplot(1,5,i+1)
                    ax.imshow(im)
                    ax.set_title(f"{classes[0]}:{final_labels[i,0]:.2f}\n{classes[1]}:{final_labels[i,1]:.2f}")
                    ax.axis("off")
                st.pyplot(fig)
                plt.close(fig)

            elif adv == "CutMix":
                cm_inp, (la, lb), lam_c = cutmix_data(images, to_one_hot(labels,2).to(device), alpha_cutmix, device=device)
                cm_labels = lam_c * la + (1 - lam_c)*lb
                fig = plt.figure(figsize=(15,4))
                for i in range(min(5, cm_inp.size(0))):
                    im = cm_inp[i].cpu().numpy().transpose(1,2,0)
                    im = (im*0.5+0.5).clip(0,1)
                    ax = fig.add_subplot(1,5,i+1)
                    ax.imshow(im)
                    ax.set_title(f"{classes[0]}:{cm_labels[i,0]:.2f}\n{classes[1]}:{cm_labels[i,1]:.2f}")
                    ax.axis("off")
                st.pyplot(fig)
                plt.close(fig)

            else:  # RICAP
                ric_inp, ric_lbl, ric_w = ricap_data(images, labels, alpha_ricap, device=device)
                fig = plt.figure(figsize=(15,4))
                for i in range(min(5, ric_inp.size(0))):
                    im = ric_inp[i].cpu().numpy().transpose(1,2,0)
                    im = (im*0.5+0.5).clip(0,1)
                    ax = fig.add_subplot(1,5,i+1)
                    ax.imshow(im)
                    txt = []
                    for kk in range(4):
                        cls_idx = ric_lbl[i][kk].item()
                        wv = ric_w[i][kk].item()
                        txt.append(f"{classes[int(cls_idx)]}({wv:.2f})")
                    ax.set_title("\n".join(txt))
                    ax.axis("off")
                st.pyplot(fig)
                plt.close(fig)

        else:
            # Для GridMask, SamplePairing или None — получаем батч из DataLoader,
            # чтобы отобразить изображения с применённым collate_fn (в частности, GridMask)
            data_iter = iter(train_loader)
            images, labels = next(data_iter)
            classes = ["Benign", "Malignant"]
            fig = plt.figure(figsize=(15, 4))
            for i in range(min(5, images.size(0))):
                im = images[i].cpu().numpy().transpose(1, 2, 0)
                im = (im*0.5 + 0.5).clip(0, 1)
                ax = fig.add_subplot(1, 5, i+1)
                # Если labels — тензор, приводим его к int
                label = int(labels[i]) if torch.is_tensor(labels) else labels[i]
                ax.set_title(classes[label])
                ax.imshow(im)
                ax.axis("off")
            st.pyplot(fig)
            plt.close(fig)

    # --- Параметры обучения ---
    st.subheader("Параметры обучения")
    epochs = st.slider("Epochs",1,30,5)
    batch_size = st.slider("Batch size",4,128,32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    st.write(f"Устройство: {device}")

    if st.button("Начать обучение"):
        with st.spinner("Загрузка данных..."):
            train_loader, val_loader, test_loader = build_dataset_and_loader(
                train_dir, val_dir, test_dir,
                manual_transform=manual_transform,
                tv_transform=tv_transform,
                batch_size=batch_size,
                advanced_aug=st.session_state["advanced_aug"],
                gridmask_params=gridmask_params if st.session_state["advanced_aug"]=="GridMask" else None,
                samplepairing_p=st.session_state.get("samplepairing_p",0.5)
            )

        progress_bar = st.progress(0)
        status_text  = st.empty()

        def progress_callback(epoch, total_epochs):
            progress_bar.progress((epoch+1)/total_epochs)
            status_text.text(f"Эпоха {epoch+1} из {total_epochs}")

        final_model, stats = train_model(
            train_loader, val_loader, device=device, epochs=epochs,
            advanced_aug=st.session_state["advanced_aug"],
            alpha_mixup=alpha_mixup,
            alpha_cutmix=alpha_cutmix,
            alpha_ricap=alpha_ricap,
            progress_callback=progress_callback
        )

        if stats is None:
            st.error("train_model не вернул статистику!")
            return
        train_losses, val_losses, train_accs, val_accs = stats

        st.subheader("Графики обучения (доверительные интервалы)")

        # Рисуем графики
        figs = plot_training_results(
            train_losses, val_losses,
            train_accs, val_accs,
            ci=True
        )
        for fig in figs:
            st.pyplot(fig)
            plt.close(fig)

        st.subheader("Точность на тесте")
        test_acc = evaluate_on_test(final_model, test_loader, device)
        st.write(f"Test Accuracy = {test_acc:.2f}%")
        st.success("Обучение завершено!")

# ------ Оптимизация Optuna -----------
def optimize_aug_params(
    train_dir, val_dir, test_dir,
    advanced_aug="None",
    use_flip=False,
    use_vflip=False,
    use_brightness=False,
    use_rotation=False,
    use_ssr=False,
    use_gaussnoise=False,
    use_gblur=False,
    use_mblur=False,
    use_sharpen=False,
    use_pixeldisp=False,
    n_trials=5
):
    import optuna
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def objective(trial: optuna.trial.Trial):
        from augmentations import make_albumentations
        from training import train_model
        from datasets import build_dataset_and_loader

        flip_prob = trial.suggest_float("flip_prob", 0.1, 0.9) if use_flip else 0.5
        vflip_prob = trial.suggest_float("vflip_prob", 0.1, 0.9) if use_vflip else 0.5
        brightness_limit = trial.suggest_float("brightness_limit", 0.1, 0.6) if use_brightness else 0.2
        rotation_limit = trial.suggest_int("rotation_limit", 0, 90) if use_rotation else 30
        shift_limit = trial.suggest_float("shift_limit", 0.0, 0.3) if use_ssr else 0.0625
        scale_limit = trial.suggest_float("scale_limit", 0.0, 0.5) if use_ssr else 0.1
        rotate_limit = trial.suggest_int("rotate_limit", 0, 90) if use_ssr else 45
        gauss_varlimit = trial.suggest_int("gauss_varlimit", 10, 100) if use_gaussnoise else 30
        gblur_limit = trial.suggest_int("gblur_limit", 3, 15) if use_gblur else 3
        mblur_limit = trial.suggest_int("mblur_limit", 3, 15) if use_mblur else 3
        sharpen_alpha = trial.suggest_float("sharpen_alpha", 0.1, 1.0) if use_sharpen else 0.3
        pixel_disp_std = trial.suggest_float("pixel_disp_std", 0.1, 5.0) if use_pixeldisp else 0.7

        alpha_mixup, alpha_cutmix, alpha_ricap = 0.2, 0.2, 0.3
        samplepairing_p = 0.5

        if advanced_aug == "MixUp":
            alpha_mixup = trial.suggest_float("alpha_mixup", 0.0, 1.0)
        elif advanced_aug == "CutMix":
            alpha_cutmix = trial.suggest_float("alpha_cutmix", 0.0, 1.0)
        elif advanced_aug == "RICAP":
            alpha_ricap = trial.suggest_float("alpha_ricap", 0.0, 1.0)
        elif advanced_aug == "GridMask":
            gridmask_dmin = trial.suggest_int("gridmask_dmin", 1, 50)
            # Гарантируем, что gridmask_dmax >= gridmask_dmin:
            gridmask_dmax = trial.suggest_int("gridmask_dmax", gridmask_dmin, 100)
            gridmask_r = trial.suggest_float("gridmask_r", 0.0, 1.0)
            gridmask_p = trial.suggest_float("gridmask_p", 0.0, 1.0)
            gridmask_params = dict(d_min=gridmask_dmin, d_max=gridmask_dmax, r=gridmask_r, p=gridmask_p)
        elif advanced_aug == "SamplePairing":
            samplepairing_p = trial.suggest_float("samplepairing_p", 0.0, 1.0)
            gridmask_params = None
        else:
            gridmask_params = None

        transform = make_albumentations(
            use_flip=use_flip, flip_prob=flip_prob,
            use_vflip=use_vflip, vflip_prob=vflip_prob,
            use_brightness=use_brightness, brightness_limit=brightness_limit,
            use_rotation=use_rotation, max_rotation=rotation_limit,
            use_shift_scale_rot=use_ssr,
            shift_limit=shift_limit, scale_limit=scale_limit, rotate_limit=rotate_limit,
            use_gaussnoise=use_gaussnoise, gauss_varlimit=gauss_varlimit,
            use_gaussianblur=use_gblur, blur_limit=gblur_limit,
            use_motionblur=use_mblur, mblur_limit=mblur_limit,
            use_sharpen=use_sharpen, sharpen_alpha=sharpen_alpha,
            use_pixeldisp=use_pixeldisp, pixel_disp_std=pixel_disp_std
        )

        train_loader, val_loader, _ = build_dataset_and_loader(
            train_dir, val_dir, test_dir,
            manual_transform=transform,
            tv_transform=None,
            batch_size=16,
            advanced_aug=advanced_aug,
            gridmask_params=gridmask_params,
            samplepairing_p=samplepairing_p
        )

        model, stats = train_model(
            train_loader, val_loader, device=device, epochs=3,
            advanced_aug=advanced_aug,
            alpha_mixup=alpha_mixup,
            alpha_cutmix=alpha_cutmix,
            alpha_ricap=alpha_ricap
        )
        if stats is None:
            return 0.0
        val_acc = stats[3][-1]
        return val_acc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    return study.best_params, study.best_value


if __name__ == "__main__":
    main()
