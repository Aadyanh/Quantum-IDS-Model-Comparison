# V3 QNN CICIDS2017 Pipeline (GPU version)

A **Quantum Neural Network (QNN)** that detects network intrusions in **CICIDS2017**.
6 qubits, **amplitude encoding** of 64 features, 4 layers of Rx/Ry/Rz + a CNOT chain, readout = <Z> of qubit 0.
The circuit is simulated **exactly on the GPU in PyTorch** (instead of one CUDA-Q call per sample), so it trains
on the whole dataset with Adam and exact gradients.

## Project Structure

```text
V3_QNN_CICIDS2017/
├── main.py                 # runs the whole pipeline
├── data/                   # CICIDS2017 .parquet files (not tracked by git)
├── results/                # model weights, metrics, plots
└── src/
    ├── config.py           # all settings in one place
    ├── preprocess.py       # loading, cleaning, de-duplication, Random Forest feature selection
    ├── split_data.py       # stratified 80/10/10 split, scaling, L2 normalisation
    ├── quantum_model.py    # GPU simulation of the 6-qubit circuit (+ self-test against a dense reference)
    ├── train_qnn.py        # Adam training with random restarts, best epoch on validation
    ├── evaluate.py         # threshold tuned on validation, metrics on the held-out test split
    └── visualize_data.py   # loss, confusion matrix, ROC, PR, metrics, feature importance plots
```

## What the pipeline does

1. **Load**: all parquet files, cleaned (inf/NaN -> 0), duplicate flows removed, labels -> Benign 0 / Attack 1.
2. **Split first**: stratified 80 / 10 / 10 train / validation / test over the FULL dataset.
3. **Fit on train only**: Random Forest picks the top 64 features, signed log, min-max scaling, L2 normalisation.
4. **Model**: amplitude-encode the 64 values, 4 layers of Rx, Ry, Rz on every qubit + CNOT chain, measure <Z0>.
   A learnable scale and bias turns <Z0> into a logit.
5. **Train**: Adam (one-cycle LR), sqrt-smoothed class weight, 3 random restarts, best epoch kept (validation loss).
6. **Evaluate**: probability threshold maximising F1 on validation; accuracy, precision, recall, F1, ROC-AUC, PR-AUC,
   confusion matrix and a per-attack-type detection table on the TEST split.

## Usage

**Local**
```bash
pip install -r ../requirements.txt
# put the dataset in V3_QNN_CICIDS2017/data/  (download links are in the Dataset section below)
cd V3_QNN_CICIDS2017
python main.py
```

**Kaggle / notebook**: attach the dataset as an input (it is found under `/kaggle/input`) and run `python main.py`
or `import main; main.main()`.

- Data needed: CICIDS2017 parquet/CSV files.
- Hardware: GPU recommended; on CPU, use `MAX_SAMPLES` in `src/config.py` (e.g. 500_000) for a quick run.
- Settings: `src/config.py`. Results are written to `results/`.

## Dataset

- **CICIDS2017**, official source: https://www.unb.ca/cic/datasets/ids-2017.html
- Loaded as parquet/CSV files that contain a `Label` column. Not included in this repository; place them in `data/` (or attach as a Kaggle input).
- Task: binary (BENIGN vs attack), about 15% attacks after de-duplication, 14 attack types.

## Model summary

| Item | Value |
|---|---|
| Type | Variational quantum neural network (QNN) |
| Qubits | 6 (64 amplitudes) |
| Features | Top 64 by Random Forest importance |
| Encoding | Amplitude encoding (L2-normalised) |
| Circuit | 4 x [RX, RY, RZ per qubit + CNOT chain] |
| Readout | <Z0> -> learnable scale and bias |
| Parameters | 74 |
| Training | Adam, OneCycleLR, 30 epochs, batch 8192, 3 restarts |
| Split | Stratified 80 / 10 / 10 |
| Baselines | None |

## Results (test split, threshold 0.6275 tuned on validation)

| Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| 0.9678 | 0.9272 | 0.8535 | 0.8889 | 0.9831 | 0.9274 |

Full report in `results/final_results.txt`.

## Notes

- Large attack classes are detected well (DDoS 0.988, DoS Hulk 0.873). Small classes (Bot, FTP-Patator, web attacks, Infiltration) are mostly missed.
- Amplitude encoding discards the overall magnitude of each sample.
- No CUDA-Q cross-check is included. The simulator is self-tested against a dense-matrix reference (`quantum_model.selftest`).
- See the root README for the comparison of all five versions.
