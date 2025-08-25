# visualization.py

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

def show_augmented_images(dataset, classes, n=4):
    """
    Возвращает фигуру (fig) с n аугментированными изображениями из датасета.
    """
    loader = DataLoader(dataset, batch_size=n, shuffle=True)
    images, labels = next(iter(loader))

    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    for i in range(n):
        # Предполагаем, что изображения нормированы mean=0.5, std=0.5
        img = images[i] * 0.5 + 0.5  # денормализация
        npimg = img.numpy().transpose((1, 2, 0))
        axes[i].imshow(np.clip(npimg, 0, 1))
        if classes and len(classes) > 0:
            axes[i].set_title(classes[labels[i]])
        axes[i].axis("off")
    fig.suptitle("Пример аугментированных изображений")
    fig.tight_layout()
    return fig


def plot_training_results(
    train_losses, val_losses,
    train_accs=None, val_accs=None,
    ci=True
):
    """
    Рисует 1 или 2 графика (Loss + Accuracy) и возвращает их в виде списка [fig_loss, fig_acc?].
    Если нет train_accs/val_accs, будет только [fig_loss].
    ci=True включает отображение доверительного интервала как ±std.
    """
    figs = []

    epochs = len(train_losses)
    x = np.arange(epochs)

    # Преобразуем в np.array
    train_losses = np.array(train_losses)
    val_losses   = np.array(val_losses)

    train_std = np.std(train_losses) if ci else 0.0
    val_std   = np.std(val_losses)   if ci else 0.0

    # --- График Loss ---
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    if ci:
        ax1.fill_between(
            x, train_losses - train_std, train_losses + train_std,
            alpha=0.1, color="blue"
        )
        ax1.fill_between(
            x, val_losses - val_std, val_losses + val_std,
            alpha=0.1, color="red"
        )
    ax1.plot(x, train_losses, label="Train Loss", color="blue", marker='o')
    ax1.plot(x, val_losses,   label="Val Loss",   color="red",  marker='o')
    ax1.set_title("Training and Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True)
    ax1.legend()
    figs.append(fig1)

    # --- График Accuracy ---
    if train_accs is not None and val_accs is not None:
        train_accs = np.array(train_accs)
        val_accs   = np.array(val_accs)
        train_acc_std = np.std(train_accs) if ci else 0.0
        val_acc_std   = np.std(val_accs)   if ci else 0.0

        fig2, ax2 = plt.subplots(figsize=(8, 4))
        if ci:
            ax2.fill_between(
                x, train_accs - train_acc_std, train_accs + train_acc_std,
                alpha=0.1, color="blue"
            )
            ax2.fill_between(
                x, val_accs - val_acc_std, val_accs + val_acc_std,
                alpha=0.1, color="red"
            )
        ax2.plot(x, train_accs, label="Train Acc", color="blue", marker='o')
        ax2.plot(x, val_accs,   label="Val Acc",   color="red",  marker='o')
        ax2.set_title("Training and Validation Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.grid(True)
        ax2.legend()
        figs.append(fig2)

    return figs
