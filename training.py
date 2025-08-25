import torch
import torch.nn as nn
import numpy as np
import logging

from models import get_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def mixup_data(inputs: torch.Tensor, labels: torch.Tensor, alpha: float = 0.2, device: str = 'cpu'):
    """
    Применяет MixUp к входным данным.
    """
    if alpha <= 0:
        return inputs, labels, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = inputs.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_inputs = lam * inputs + (1 - lam) * inputs[index, :]
    labels_a, labels_b = labels, labels[index]
    return mixed_inputs, (labels_a, labels_b), lam


def mixup_criterion(criterion, outputs, labels_tuple, lam):
    labels_a, labels_b = labels_tuple
    return lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)


def cutmix_data(inputs: torch.Tensor, labels: torch.Tensor, alpha: float = 0.2, device: str = 'cpu'):
    """
    Применяет CutMix к входным данным.
    """
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


def cutmix_criterion(criterion, outputs, labels_tuple, lam):
    labels_a, labels_b = labels_tuple
    return lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)


def ricap_data(inputs: torch.Tensor, labels: torch.Tensor, alpha: float = 0.3, device: str = 'cpu'):
    """
    Применяет RICAP к входным данным.
    """
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
            idx = np.random.randint(0, B)
            x_k = np.random.randint(0, W - w_[k] + 1)
            y_k = np.random.randint(0, H - h_[k] + 1)
            x1, x2 = x_k, x_k + w_[k]
            y1, y2 = y_k, y_k + h_[k]
            if k == 0:
                new_inputs[b, :, 0:h_[k], 0:w_[k]] = inputs[idx, :, y1:y2, x1:x2]
            elif k == 1:
                new_inputs[b, :, 0:h_[k], w_[0]:w_[0] + w_[k]] = inputs[idx, :, y1:y2, x1:x2]
            elif k == 2:
                new_inputs[b, :, h_[0]:h_[0] + h_[k], 0:w_[k]] = inputs[idx, :, y1:y2, x1:x2]
            else:
                new_inputs[b, :, h_[1]:h_[1] + h_[k], w_[2]:w_[2] + w_[k]] = inputs[idx, :, y1:y2, x1:x2]
            new_labels[b, k] = labels[idx]
            weights[b, k] = w_[k] * h_[k] / float(W * H)
    return new_inputs, new_labels, weights


def ricap_criterion(criterion, outputs, label_parts, weights):
    B = outputs.size(0)
    loss = 0.0
    for k in range(4):
        l_k = criterion(outputs, label_parts[:, k])
        loss += (weights[:, k] * l_k).mean()
    return loss


def train_model(
        train_loader, val_loader, device, epochs: int = 5,
        advanced_aug: str = "None", alpha_mixup: float = 0.2, alpha_cutmix: float = 0.2, alpha_ricap: float = 0.3,
        progress_callback=None
):
    """
    Обучает модель и возвращает обученную модель вместе со статистикой обучения.

    Args:
        train_loader: DataLoader для обучающей выборки.
        val_loader: DataLoader для валидационной выборки.
        device: Устройство для обучения.
        epochs: Число эпох.
        advanced_aug: Тип продвинутой аугментации.
        alpha_mixup: Параметр MixUp.
        alpha_cutmix: Параметр CutMix.
        alpha_ricap: Параметр RICAP.
        progress_callback: Функция обратного вызова для обновления прогресса.

    Returns:
        model: Обученная модель.
        stats: Кортеж (train_losses, val_losses, train_accs, val_accs)
    """
    logging.info("Начало обучения модели")
    model = get_model(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for ep in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            if advanced_aug == "MixUp":
                labels_oh = torch.nn.functional.one_hot(labels, 2).float().to(device)
                mixed_inputs, (la, lb), lam_m = mixup_data(inputs, labels_oh, alpha=alpha_mixup, device=device)
                outputs = model(mixed_inputs)
                loss = mixup_criterion(criterion, outputs, (la, lb), lam_m)
            elif advanced_aug == "CutMix":
                labels_oh = torch.nn.functional.one_hot(labels, 2).float().to(device)
                cm_inputs, (la, lb), lam_c = cutmix_data(inputs, labels_oh, alpha=alpha_cutmix, device=device)
                outputs = model(cm_inputs)
                loss = cutmix_criterion(criterion, outputs, (la, lb), lam_c)
            elif advanced_aug == "RICAP":
                ric_inputs, ric_labels, ric_weights = ricap_data(inputs, labels, alpha=alpha_ricap, device=device)
                outputs = model(ric_inputs)
                loss = ricap_criterion(criterion, outputs, ric_labels,
                                       ric_weights) if ric_weights is not None else criterion(outputs, labels)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_acc = 100.0 * correct_train / total_train
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)

        # Валидация
        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs_val, labels_val in val_loader:
                inputs_val = inputs_val.to(device)
                labels_val = labels_val.to(device)
                outputs_val = model(inputs_val)
                loss_val = criterion(outputs_val, labels_val)
                val_running_loss += loss_val.item()
                _, predicted_val = torch.max(outputs_val.data, 1)
                total_val += labels_val.size(0)
                correct_val += (predicted_val == labels_val).sum().item()
        epoch_val_loss = val_running_loss / len(val_loader)
        epoch_val_acc = 100.0 * correct_val / total_val
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)

        logging.info(
            f"Эпоха {ep + 1}/{epochs} - Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%, Val Acc: {epoch_val_acc:.2f}%")

        if progress_callback is not None:
            progress_callback(ep, epochs)

    logging.info("Обучение завершено")
    return model, (train_losses, val_losses, train_accs, val_accs)


def evaluate_on_test(model, test_loader, device):
    """
    Оценивает точность модели на тестовом наборе.
    """
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_acc = 100.0 * correct / total
    return test_acc
