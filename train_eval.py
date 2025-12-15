"""Training and evaluation loops."""

from __future__ import annotations
import numpy as np
import torch
import sklearn.metrics


def train_one_epoch(model, optimizer, criterion, dataloader, device: str):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


def evaluate(model, criterion, dataloader, device: str, n_classes: int = 3):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    class_correct = np.zeros(n_classes, dtype=float)
    class_total = np.zeros(n_classes, dtype=float)

    y_true, y_pred = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device, dtype=torch.float)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

            for c in range(n_classes):
                class_correct[c] += ((predicted == c) & (labels == c)).sum().item()
                class_total[c] += (labels == c).sum().item()

    epoch_loss = running_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)

    # avoid divide-by-zero
    class_accuracy = np.divide(class_correct, class_total, out=np.zeros_like(class_correct), where=class_total != 0)
    confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))

    return epoch_loss, epoch_acc, class_accuracy, confusion_matrix
