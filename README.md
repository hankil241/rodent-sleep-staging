# Rodent Sleep Stage Classification (Cross- & Within-Subject)

This repository contains PyTorch implementations for:
- Cross-subject sleep stage classification
- Within-subject sleep stage classification

Models supported:
- 1D-CNN
- 2D-CNN (SSANN)
- CNN + BiLSTM (DeepSleepNet-style)

## Structure
├── cross_subject.py        # LOSO / cross-subject evaluation
├── within_subject.py      # within-subject evaluation
├── mixture_zscore.py      # mixture z-scoring
├── train_eval.py          # training & evaluation loops
├── device_utils.py        # CUDA / device selection
├── run_cross_subject.py   # experiment entry point
├── run_within_subject.py
└── Model.py               # model definitions (not include, search original published paper for code)

```bash
python run_cross_subject.py
python run_within_subject.py
```