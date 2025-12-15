"""Entry point for within-subject experiment."""

from __future__ import annotations
import os
import pickle
import numpy as np

from device_utils import print_cuda_info, get_device
from within_subject import within_validate_model


def main():
    print_cuda_info()
    device = get_device()
    print(f"Using {device} device")

    model_type = "1D-CNN"  # '1D-CNN', '2D-CNN', 'CNN+BiLSTM'

    base_path = "Result_metrics/within"
    if model_type == "1D-CNN":
        model_folder = os.path.join(base_path, "1D-CNN")
    elif model_type == "2D-CNN":
        model_folder = os.path.join(base_path, "2D-CNN")
    elif model_type == "CNN+BiLSTM":
        model_folder = os.path.join(base_path, "CNN+BiLSTM")
    else:
        raise ValueError("Model type not recognized!")

    os.makedirs(model_folder, exist_ok=True)

    if model_type == "2D-CNN":
        all_data = np.load("path/to/your/epoched_spectrogram_data.npy")
        print(all_data.shape)
        all_labels = np.load("path/to/your/label_data.npy").squeeze() - 1
        print(all_labels.shape)

        metrics = within_validate_model(all_data, None, all_labels, device, model_type=model_type, isZscore=False, isMixture=False)
    else:
        all_eeg_data = np.load("path/to/your/epoched_eeg_data.npy")
        all_emg_data = np.load("path/to/your/epoched_emg_data.npy")
        print(all_eeg_data.shape)
        print(all_emg_data.shape)

        all_labels = np.load("path/to/your/label_data.npy").squeeze() - 1
        print(all_labels.shape)

        metrics = within_validate_model(all_eeg_data, all_emg_data, all_labels, device, model_type=model_type, isZscore=False, isMixture=False)

    with open(os.path.join(model_folder, "matrix_no_norm.pkl"), "wb") as f:
        pickle.dump(metrics, f)

    print(f"Metrics saved for {model_type} in {model_folder}")


if __name__ == "__main__":
    main()
