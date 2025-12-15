"""Cross-subject validation logic (train on one mouse, validate on others).

You must provide a local `Model.py` with:
  - CNN1DModel, SSANN, DeepSleepNet
  - Datasets: CNN1D, SSDataset, CNNBiLSTM
"""

from __future__ import annotations

import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

import Model as Model  # noqa: F401

from mixture_zscore import mixture_zscore, cal_mean_std, apply_mixture_zscore, compute_class_weights
from train_eval import train_one_epoch, evaluate


def cross_validate_model(
    all_eeg_data,
    all_emg_data,
    mice_labels,
    device: str,
    model_type: str = "1D-CNN",
    isZscore: bool = False,
    isMixture: bool = False,
    num_epochs: int = 25,
    batch_size: int = 128,
):
    # Define data size based on model type
    if model_type == "2D-CNN":
        data_size = all_eeg_data.shape[-2:]
        all_emg_data = None
    else:
        data_size = all_eeg_data.shape[-1]

    all_mouse_metrics = []

    # Loop through each mouse for training
    for train_mouse_idx in range(len(all_eeg_data)):
        print(f"Training on Mouse {train_mouse_idx + 1}")

        if model_type == "2D-CNN":
            train_mouse_data = all_eeg_data[train_mouse_idx]
            shape = (1, *data_size)
            train_data = train_mouse_data.reshape(-1, *shape)
        else:
            eeg_train_mouse_data = all_eeg_data[train_mouse_idx]
            emg_train_mouse_data = all_emg_data[train_mouse_idx]
            shape = (data_size,)
            eeg_train_data = eeg_train_mouse_data.reshape(-1, *shape)
            emg_train_data = emg_train_mouse_data.reshape(-1, *shape)

        train_labels = mice_labels[train_mouse_idx].reshape(-1)
        original_train_labels = train_labels.copy()

        # SMOTE
        if model_type == "2D-CNN":
            smote = SMOTE(sampling_strategy="auto", random_state=42)
            flat = train_data.reshape(-1, train_data.shape[-2] * train_data.shape[-1])
            flat, train_labels = smote.fit_resample(flat, train_labels)
            train_data = flat.reshape(-1, *shape)
        else:
            eeg_smote = SMOTE(sampling_strategy="auto", random_state=42)
            eeg_train_data, eeg_train_labels = eeg_smote.fit_resample(eeg_train_data, train_labels)

            emg_smote = SMOTE(sampling_strategy="auto", random_state=42)
            emg_train_data, emg_train_labels = emg_smote.fit_resample(emg_train_data, train_labels)

            if np.array_equal(eeg_train_labels, emg_train_labels):
                train_labels = eeg_train_labels

        unique_train, counts_train = np.unique(train_labels, return_counts=True)
        print(f"  Class distribution for Training Data: {dict(zip(unique_train, counts_train))}")

        # Z-score
        scaler = None
        if isZscore:
            scaler = StandardScaler()
            if model_type == "2D-CNN":
                flat = train_data.reshape(-1, train_data.shape[-2] * train_data.shape[-1])
                flat = scaler.fit_transform(flat)
                train_data = flat.reshape(train_data.shape)
            else:
                eeg_train_data = scaler.fit_transform(eeg_train_data)
                emg_train_data = scaler.fit_transform(emg_train_data)

        # Mixture z-score
        if isMixture:
            if model_type == "2D-CNN":
                flat_train = train_data.reshape(-1, train_data.shape[-2] * train_data.shape[-1])
                z_train, train_weights = mixture_zscore(original_train_labels, flat_train, train_labels)
                train_data = z_train.reshape(train_data.shape)
            else:
                eeg_train_data, eeg_weights = mixture_zscore(original_train_labels, eeg_train_data, train_labels)
                emg_train_data, emg_weights = mixture_zscore(original_train_labels, emg_train_data, train_labels)

        # Init model/dataset
        if model_type == "1D-CNN":
            model = Model.CNN1DModel().to(device)
            train_dataset = Model.CNN1D(
                eeg_train_data.reshape(-1, 1, data_size),
                emg_train_data.reshape(-1, 1, data_size),
                train_labels,
            )
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.0001)

        elif model_type == "2D-CNN":
            model = Model.SSANN().to(device)
            train_dataset = Model.SSDataset(train_data, train_labels)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=0.015, momentum=0.9)

        elif model_type == "CNN+BiLSTM":
            model = Model.DeepSleepNet(
                n_outputs=3,
                return_feats=False,
                n_chans=2,
                chs_info=None,
                n_times=1280,
                input_window_seconds=2.5,
                sfreq=512,
                n_classes=None,
            ).to(device)
            train_dataset = Model.CNNBiLSTM(
                eeg_train_data.reshape(-1, 1, data_size),
                emg_train_data.reshape(-1, 1, data_size),
                train_labels,
            )
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.05)

        else:
            raise ValueError("Model type not recognized!")

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Train
        initial_learning_rate = 0.015
        for epoch in range(num_epochs):
            if model_type == "2D-CNN":
                lr = initial_learning_rate * (0.85 ** epoch)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr
            train_loss, train_acc = train_one_epoch(model, optimizer, criterion, train_loader, device)
            print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f}")

        # Validate on other mice
        for val_mouse_idx in range(len(all_eeg_data)):
            if val_mouse_idx == train_mouse_idx:
                continue

            print(f"  Validating on Mouse {val_mouse_idx + 1}")

            if model_type == "2D-CNN":
                val_mouse_data = all_eeg_data[val_mouse_idx]
            else:
                eeg_val_mouse_data = all_eeg_data[val_mouse_idx]
                emg_val_mouse_data = all_emg_data[val_mouse_idx]

            val_mouse_labels = mice_labels[val_mouse_idx]

            val_metrics = []
            for day in range(5):
                mean_var_labels = val_mouse_labels[day].reshape(-1)

                if model_type == "2D-CNN":
                    mean_var_data = val_mouse_data[day].reshape(-1, val_mouse_data.shape[-2] * val_mouse_data.shape[-1])
                    mu, sigma2 = cal_mean_std(mean_var_data, mean_var_labels)
                else:
                    eeg_mean_var_data = eeg_val_mouse_data[day].reshape(-1, *shape)
                    emg_mean_var_data = emg_val_mouse_data[day].reshape(-1, *shape)
                    eeg_mu, eeg_sigma2 = cal_mean_std(eeg_mean_var_data, mean_var_labels)
                    emg_mu, emg_sigma2 = cal_mean_std(emg_mean_var_data, mean_var_labels)

                val_days_indices = [i for i in range(5) if i != day]
                if model_type == "2D-CNN":
                    val_data = np.concatenate([val_mouse_data[i].reshape(-1, *shape) for i in val_days_indices], axis=0)
                else:
                    eeg_val_data = np.concatenate([eeg_val_mouse_data[i].reshape(-1, *shape) for i in val_days_indices], axis=0)
                    emg_val_data = np.concatenate([emg_val_mouse_data[i].reshape(-1, *shape) for i in val_days_indices], axis=0)

                val_labels = np.concatenate([val_mouse_labels[i].reshape(-1) for i in val_days_indices], axis=0)

                unique_val, counts_val = np.unique(val_labels, return_counts=True)
                print(f"    Day {day + 1} Class distribution for Validation Data: {dict(zip(unique_val, counts_val))}")

                # Apply Z-score using training scaler if enabled
                if isZscore and scaler is not None:
                    if model_type == "2D-CNN":
                        flat = val_data.reshape(-1, val_data.shape[-2] * val_data.shape[-1])
                        flat = scaler.transform(flat)
                        val_data = flat.reshape(val_data.shape)
                    else:
                        eeg_val_data = scaler.transform(eeg_val_data)
                        emg_val_data = scaler.transform(emg_val_data)

                # Apply Mixture z-score using folded stats if enabled
                if isMixture:
                    if model_type == "2D-CNN":
                        flat = val_data.reshape(-1, val_data.shape[-2] * val_data.shape[-1])
                        flat = apply_mixture_zscore(flat, train_weights, mu, sigma2)
                        val_data = flat.reshape(val_data.shape)
                    else:
                        eeg_val_data = apply_mixture_zscore(eeg_val_data, eeg_weights, eeg_mu, eeg_sigma2)
                        emg_val_data = apply_mixture_zscore(emg_val_data, emg_weights, emg_mu, emg_sigma2)

                # Dataset
                if model_type == "1D-CNN":
                    val_dataset = Model.CNN1D(
                        eeg_val_data.reshape(-1, 1, data_size),
                        emg_val_data.reshape(-1, 1, data_size),
                        val_labels,
                    )
                elif model_type == "2D-CNN":
                    val_dataset = Model.SSDataset(val_data, val_labels)
                elif model_type == "CNN+BiLSTM":
                    val_dataset = Model.CNNBiLSTM(
                        eeg_val_data.reshape(-1, 1, data_size),
                        emg_val_data.reshape(-1, 1, data_size),
                        val_labels,
                    )
                else:
                    raise ValueError("Model type not recognized!")

                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                val_loss, val_acc, class_acc, confusion_mat = evaluate(model, criterion, val_loader, device)

                print(f"      Day {day + 1} Validation Loss = {val_loss:.4f}, Validation Acc = {val_acc:.4f}")
                print(f"      Class Accuracies: REM = {class_acc[0]:.4f}, Wake = {class_acc[1]:.4f}, NREM = {class_acc[2]:.4f}")

                val_metrics.append((val_loss, val_acc, class_acc, confusion_mat))

            avg_val_loss = np.mean([m[0] for m in val_metrics])
            avg_val_acc = np.mean([m[1] for m in val_metrics])
            avg_class_acc = np.mean([m[2] for m in val_metrics], axis=0)

            print(f"    Average Validation Loss for Mouse {val_mouse_idx + 1}: {avg_val_loss:.4f}")
            print(f"    Average Validation Accuracy for Mouse {val_mouse_idx + 1}: {avg_val_acc:.4f}")
            print(f"    Average Class Accuracies for Mouse {val_mouse_idx + 1}: REM = {avg_class_acc[0]:.4f}, Wake = {avg_class_acc[1]:.4f}, NREM = {avg_class_acc[2]:.4f}")

            # store last confusion_mat from the loop (matches your original behavior)
            all_mouse_metrics.append((avg_val_loss, avg_val_acc, avg_class_acc, val_metrics[-1][3]))

        # Summary across validations accumulated so far
        avg_val_loss = np.mean([m[0] for m in all_mouse_metrics])
        avg_val_acc = np.mean([m[1] for m in all_mouse_metrics])
        avg_class_acc = np.mean([m[2] for m in all_mouse_metrics], axis=0)

        print(f"Average Validation Loss: {avg_val_loss:.4f}")
        print(f"Average Validation Accuracy: {avg_val_acc:.4f}")
        print(f"Average Class Accuracies: REM = {avg_class_acc[0]:.4f}, Wake = {avg_class_acc[1]:.4f}, NREM = {avg_class_acc[2]:.4f}")

    overall_avg_val_loss = np.mean([m[0] for m in all_mouse_metrics])
    overall_avg_val_acc = np.mean([m[1] for m in all_mouse_metrics])
    overall_avg_class_acc = np.mean([m[2] for m in all_mouse_metrics], axis=0)

    print(f"Overall Average Validation Loss: {overall_avg_val_loss:.4f}")
    print(f"Overall Average Validation Accuracy: {overall_avg_val_acc:.4f}")
    print(f"Overall Average Class Accuracies: REM = {overall_avg_class_acc[0]:.4f}, Wake = {overall_avg_class_acc[1]:.4f}, NREM = {overall_avg_class_acc[2]:.4f}")

    return all_mouse_metrics
