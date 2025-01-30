import torch
import torch.nn as nn
import numpy as np

# MixUp, CutMix, RICAP
def mixup_data(inputs, labels, alpha=0.2, device='cpu'):
    if alpha <= 0:
        return inputs, labels, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = inputs.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_inputs = lam * inputs + (1 - lam)*inputs[index,:]
    labels_a, labels_b = labels, labels[index]
    return mixed_inputs, (labels_a, labels_b), lam

def mixup_criterion(criterion, outputs, labels_tuple, lam):
    labels_a, labels_b = labels_tuple
    return lam * criterion(outputs, labels_a) + (1-lam)*criterion(outputs, labels_b)

def cutmix_data(inputs, labels, alpha=0.2, device='cpu'):
    if alpha <= 0:
        return inputs, labels, 1.0
    lam = np.random.beta(alpha, alpha)
    B, C, H, W = inputs.size()
    index = torch.randperm(B).to(device)
    inputs_shuf = inputs[index,:]
    labels_a, labels_b = labels, labels[index]

    rx = np.random.randint(W)
    ry = np.random.randint(H)
    cut_w = int(W * np.sqrt(1 - lam))
    cut_h = int(H * np.sqrt(1 - lam))

    x1 = np.clip(rx - cut_w//2, 0, W)
    x2 = np.clip(rx + cut_w//2, 0, W)
    y1 = np.clip(ry - cut_h//2, 0, H)
    y2 = np.clip(ry + cut_h//2, 0, H)

    inputs[:, :, y1:y2, x1:x2] = inputs_shuf[:, :, y1:y2, x1:x2]

    area = (y2-y1)*(x2-x1)
    lam = 1 - area/float(W*H)
    return inputs, (labels_a, labels_b), lam

def cutmix_criterion(criterion, outputs, labels_tuple, lam):
    labels_a, labels_b = labels_tuple
    return lam * criterion(outputs, labels_a) + (1-lam)*criterion(outputs, labels_b)

def ricap_data(inputs, labels, alpha=0.3, device='cpu'):
    if alpha <= 0:
        return inputs, labels, None
    B, C, H, W = inputs.shape
    cx = np.random.randint(1, W)
    cy = np.random.randint(1, H)
    w_ = [cx, W-cx, cx, W-cx]
    h_ = [cy, cy, H-cy, H-cy]
    new_inputs = torch.zeros_like(inputs)
    new_labels = torch.zeros((B,4), dtype=torch.long, device=device)
    weights    = torch.zeros((B,4), dtype=torch.float, device=device)

    for b in range(B):
        for k in range(4):
            idx = np.random.randint(0,B)
            x_k = np.random.randint(0, W-w_[k]+1)
            y_k = np.random.randint(0, H-h_[k]+1)
            x1, x2 = x_k, x_k + w_[k]
            y1, y2 = y_k, y_k + h_[k]

            if k==0:
                new_inputs[b, :, 0:h_[k], 0:w_[k]] = inputs[idx, :, y1:y2, x1:x2]
            elif k==1:
                new_inputs[b, :, 0:h_[k], w_[0]:w_[0]+w_[k]] = inputs[idx, :, y1:y2, x1:x2]
            elif k==2:
                new_inputs[b, :, h_[0]:h_[0]+h_[k], 0:w_[k]] = inputs[idx, :, y1:y2, x1:x2]
            else:
                new_inputs[b, :, h_[1]:h_[1]+h_[k], w_[2]:w_[2]+w_[k]] = inputs[idx, :, y1:y2, x1:x2]

            new_labels[b,k] = labels[idx]
            weights[b,k] = w_[k]*h_[k]/float(W*H)

    return new_inputs, new_labels, weights

def ricap_criterion(criterion, outputs, label_parts, weights):
    B = outputs.size(0)
    loss = 0.0
    for k in range(4):
        l_k = criterion(outputs, label_parts[:,k])
        loss += (weights[:,k]*l_k).mean(dim=0)
    return loss

def train_model(
    train_loader, val_loader, device, epochs=5,
    advanced_aug="None", alpha_mixup=0.2, alpha_cutmix=0.2, alpha_ricap=0.3,
):
    """
    Основная функция обучения.
    """
    from models import get_model
    model = get_model(num_classes=2).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    train_losses = []
    val_losses   = []
    train_accs   = []
    val_accs     = []

    for ep in range(epochs):
        model.train()
        run_loss = 0.0
        correct_train = 0
        total_train   = 0

        for (inp, lbl) in train_loader:
            inp = inp.to(device)
            lbl = lbl.to(device)

            optimizer.zero_grad()

            if advanced_aug == "MixUp":
                lbl_oh = torch.nn.functional.one_hot(lbl, 2).float().to(device)
                mixed_inp, (la, lb), lam_m = mixup_data(inp, lbl_oh, alpha=alpha_mixup, device=device)
                out = model(mixed_inp)
                loss = mixup_criterion(criterion, out, (la, lb), lam_m)

            elif advanced_aug == "CutMix":
                lbl_oh = torch.nn.functional.one_hot(lbl, 2).float().to(device)
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
                out = model(inp)
                loss = criterion(out, lbl)

            loss.backward()
            optimizer.step()
            run_loss += loss.item()

            _, predicted = torch.max(out.data, 1)
            total_train += lbl.size(0)
            correct_train += (predicted == lbl).sum().item()

        train_loss = run_loss / len(train_loader)
        train_acc  = 100.0 * correct_train / total_train

        # Validation
        model.eval()
        val_run_loss = 0.0
        val_correct  = 0
        val_total    = 0
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
        val_acc  = 100.0 * val_correct / val_total

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"[Epoch {ep+1}/{epochs}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

    return model, (train_losses, val_losses, train_accs, val_accs)

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
    test_acc = 100.0 * correct / total
    return test_acc
