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
└── Model.py               # model definitions (not included)
```

```bash
python run_cross_subject.py
python run_within_subject.py
```
