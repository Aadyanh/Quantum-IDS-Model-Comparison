# Quantum-IDS-Model-Comparison

Five network intrusion detection models compared across four public datasets: four quantum / hybrid models
and one classical baseline. All quantum circuits are simulated exactly on the GPU with PyTorch.

| Version | Model | Dataset | Qubits | Encoding |
|---|---|---|---|---|
| [V1](V1_QNN_NSLKDD) | Variational QNN | NSL-KDD | 8 | Angle |
| [V2](V2_QSVM_UNSW) | Quantum kernel SVM (QSVM) | UNSW-NB15 | 6 | Angle |
| [V3](V3_QNN_CICIDS2017) | Variational QNN | CICIDS2017 | 6 | Amplitude (64 features) |
| [V4](V4_HQNN_TONIOT) | Hybrid classical-quantum NN | TON_IoT | 16 | Angle (learned) |
| [V5](V5_Classical_SVM_NSLKDD) | Classical RBF SVM | NSL-KDD | none | none |

## Datasets (not included, download from the sources)

| Dataset | Used in | Source |
|---|---|---|
| NSL-KDD | V1, V5 | https://www.unb.ca/cic/datasets/nsl.html |
| UNSW-NB15 | V2 | https://research.unsw.edu.au/projects/unsw-nb15-dataset |
| CICIDS2017 | V3 | https://www.unb.ca/cic/datasets/ids-2017.html |
| TON_IoT | V4 | https://research.unsw.edu.au/projects/toniot-datasets |

Please follow each dataset's license and citation requirements.

## Headline results (held-out test, threshold tuned on validation)

| Version | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|
| V1 QNN | 0.8432 | 0.9585 | 0.9602 |
| V2 QSVM | 0.8401 | 0.9948 | 0.8743 |
| V3 QNN | 0.8889 | 0.9831 | 0.9274 |
| V4 HQNN | 0.9955 | 0.9988 | 0.9996 |
| V5 RBF SVM (tuned) | 0.8454 | 0.9508 | 0.9477 |

Scores across datasets are not directly comparable. Only V1 and V5 share data, split and features.
V4 uses far more input features and an easier dataset.

## Setup

```bash
pip install -r requirements.txt
cd V1_QNN_NSLKDD   # or any other version
python main.py
```

Datasets are not in the repo: download them (links above) into each version's `data/` folder,
or attach them as inputs when running on Kaggle (`/kaggle/input` is searched automatically).

Each version has its own README with structure, dataset details, model summary and results.

## Hardware

| Version | CPU only | GPU |
|---|---|---|
| V1 | Works | Faster |
| V2 | Slow on the full 2M rows; set `MAX_SAMPLES` for a quick run | Recommended |
| V3 | Slow on the full 2M rows; set `MAX_SAMPLES` for a quick run | Recommended |
| V4 | Very slow (16-qubit simulation); reduce `N_QUBITS` / `MAX_SAMPLES` | Strongly recommended |
| V5 | Intended (about 30 s per fit) | Not used |

## Notes

- CUDA-Q cross-checks exist for V1, V2 and V4 but were not run for the saved results.
- The simulated product-state circuits (V1, V2) are classically simulable; no quantum advantage is claimed.
