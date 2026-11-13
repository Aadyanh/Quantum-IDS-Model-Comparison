# V1 Quantum Intrusion Classifier - NSL-KDD (GPU version)

A **variational quantum classifier** for network intrusion detection on **NSL-KDD**: 8 features -> 8 qubits,
angle encoding `RY(pi * x)`, blocks of `[CNOT chain + RY(theta) layer]`, final CNOT chain, readout of `<Z>`.
The circuit only uses RY and CNOT, so it is one real orthogonal 256 x 256 matrix per parameter setting. It is simulated
**exactly on the GPU in PyTorch** (instead of one CUDA-Q call per sample), trained with Adam and exact gradients on the
**whole** training file. CUDA-Q is used afterwards to check the simulator against a CUDA-Q kernel of the same circuit.

## Project Structure

```text
V1_QNN_NSLKDD/
├── main.py                    # runs the whole pipeline
├── data/                      # KDDTrain / KDDTest (.parquet, .txt or .csv) - not tracked by git
├── results/                   # model weights, metrics, plots
└── src/
    ├── config.py              # all settings in one place
    ├── preprocess.py          # reads parquet/csv/txt, detects columns and the label column
    ├── split_data.py          # train/validation split, target encoding, RF feature selection, scaling
    ├── quantum_encoder.py     # angle encoding
    ├── quantum_classifier.py  # GPU circuit simulator (QNN) + CUDA-Q check
    ├── baselines.py           # classical logistic regression and MLP on the same features
    ├── objective.py           # scoring and feature-sensitivity helpers
    ├── train_binary.py        # training (Adam, random restarts, best epoch on validation)
    ├── evaluate.py            # threshold tuned on validation, metrics, per-class detection
    ├── visualize_data.py      # plots
    └── cudaq_verify.py        # standalone CUDA-Q circuit check
```

## What the pipeline does

1. **Load** KDDTrain and KDDTest (the official split; the test file contains 17 attack types never seen in training).
2. **Split**: KDDTrain -> 85% train / 15% validation (stratified by attack family). KDDTest is only used for testing.
3. **Fit on train only**: categorical columns -> attack rate, Random Forest picks the top 8 features, log + min-max scaling to [0, 1].
4. **Train** the improved QNN (4 layers, all-qubit readout), the original-style QNN (2 layers, mean of Z0..Z3) and two
   classical baselines. QNNs use 3 random restarts; the best one on validation is kept.
5. **Evaluate**: threshold maximising F1 on validation; accuracy, precision, recall, F1, MCC, ROC-AUC, PR-AUC on validation
   and on the official test set, detection rate per attack family, seen vs. novel attacks, and the share of each model's
   sensitivity per feature (shows which qubits/features a model actually uses).

## Usage

**Local**
```bash
pip install -r ../requirements.txt
# put the dataset in V1_QNN_NSLKDD/data/  (download links are in the Dataset section below)
cd V1_QNN_NSLKDD
python main.py
```

**Kaggle / notebook**: attach the dataset as an input (it is found under `/kaggle/input`) and run `python main.py`
or `import main; main.main()`.

- Data needed: NSL-KDD files (KDDTrain+, KDDTest+).
- Hardware: GPU recommended; CPU also works (8 qubits is small).
- Settings: `src/config.py`. Results are written to `results/`.

## Dataset

- **NSL-KDD** (KDDTrain+ / KDDTest+), official source: https://www.unb.ca/cic/datasets/nsl.html
- Kaggle mirror: https://www.kaggle.com/datasets/hassan06/nslkdd
- Not included in this repository. Download it and place the files in `data/` (or attach it as a Kaggle input).
- Task: binary (normal vs attack). KDDTrain+ 125,973 rows, KDDTest+ 22,544 rows (17 attack types unseen in training).

## Model summary

| Item | Value |
|---|---|
| Type | Variational quantum classifier (QNN) |
| Qubits | 8 (one feature per qubit) |
| Features | Top 8 by Random Forest importance |
| Encoding | Angle encoding, RY(pi * x) |
| Circuit | 4 x [CNOT chain + RY layer] + final CNOT chain |
| Readout | <Z> on all 8 qubits -> Linear(8 -> 1) |
| Parameters | 41 |
| Training | Adam, OneCycleLR, 60 epochs, batch 2048, 3 restarts |
| Baselines | Logistic regression, MLP 64-64 |

## Results (KDDTest+, threshold tuned on validation)

| Model | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| QNN improved | 0.8349 | 0.8432 | 0.9585 |
| QNN original-style | 0.7488 | 0.7417 | 0.8777 |
| Logistic regression | 0.7610 | 0.7567 | 0.9506 |
| MLP 64-64 | 0.8608 | 0.8636 | 0.9673 |

Full tables are in `results/final_results.txt` and `results/results.json`.

## Notes

- The quantum circuit is simulated exactly in PyTorch. `src/cudaq_verify.py` can cross-check it against CUDA-Q (`pip install cuda-quantum-cu12`); this check was not run for the saved results.
- V5 uses the same data, split and features, so V1 and V5 are directly comparable.
- See the root README for the comparison of all five versions.
