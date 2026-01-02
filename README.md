# Rodent Sleep Stage Classification (Cross- & Within-Subject)

This repository contains PyTorch implementations for:
- Cross-subject sleep stage classification
- Within-subject sleep stage classification

Model architectures implemented in this file are based on previously
published studies, including:
- 1D-CNN (Conventional Deep 1D-CNN): Yildirim et al., 2019, Int. J. Environ. Res. Public Health
- 2D-CNN (AccuSleep-SSANN): Barger et al., 2019, PLoS ONE
- CNN + BiLSTM (DeepSleepNet): Supratak et al., 2017, IEEE TNSRE

## Structure

```text
├── cross_subject.py        # LOSO / cross-subject evaluation
├── within_subject.py      # within-subject evaluation
├── mixture_zscore.py      # mixture z-scoring
├── train_eval.py          # training & evaluation loops
├── device_utils.py        # CUDA / device selection
├── run_cross_subject.py   # experiment entry point
├── run_within_subject.py
└── Model.py               # model definitions
└── Result_metrics/
  └── within/ # Saved evaluation metrics (pickle format)
  └── cross/ # Saved evaluation metrics (pickle format)
```

## Data Requirements

Depending on the selected model, the following inputs are required.
Data used in this study is available here: https://osf.io/py5eb/ (Barger et al., AccuSleep)

### 1D-CNN and CNN + BiLSTM
- Epoched EEG data: `epoched_eeg_data.npy`
- Epoched EMG data: `epoched_emg_data.npy`
- Sleep stage labels: `label_data.npy`

### 2D-CNN (SSANN)
- Epoched spectrogram data: `epoched_spectrogram_data.npy`
- Sleep stage labels: `label_data.npy`

All input files are expected to be NumPy `.npy` arrays. Labels are internally converted to zero-based indexing.

---

## Running Experiments

### Within-Subject Evaluation (similar for Cross-Subject)

Select the model type in `run_within_subject.py`:

```python
model_type = "1D-CNN"  # '1D-CNN', '2D-CNN', 'CNN+BiLSTM'

python run_cross_subject.py
python run_within_subject.py
```
