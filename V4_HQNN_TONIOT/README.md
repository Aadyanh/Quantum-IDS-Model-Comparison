# V4 Hybrid QNN - TON_IoT (GPU version)

A **hybrid classical-quantum neural network** for network intrusion detection on **TON_IoT**.
A classical MLP maps all input features to 16 angles, a 16-qubit variational circuit processes them,
and a linear head produces the prediction. The circuit is simulated **exactly on the GPU in PyTorch**
with autograd gradients (no parameter-shift circuits).

## Project Structure

```text
V4_HQNN_TONIOT/
├── main.py                 # runs the whole pipeline
├── data/                   # put train_test_network.csv here (not tracked by git)
├── results/                # model weights, metrics, plots
└── src/
    ├── config.py           # all settings in one place
    ├── preprocess.py       # loading, encoding, quantile scaling, 80/10/10 split
    ├── hybrid_model.py     # QuantumLayer (GPU simulator) + HybridQNN + self-test
    ├── train.py            # Adam training, best epoch on validation
    ├── evaluate.py         # threshold tuned on validation, metrics on test split
    ├── visualize_data.py   # ROC / PR / confusion matrix / loss plots
    └── cudaq_worker.py     # optional CUDA-Q cross-check in a separate process
```

## Dataset

- **TON_IoT** network dataset, file `train_test_network.csv` (Train_Test_datasets), official source: https://research.unsw.edu.au/projects/toniot-datasets
- Not included in this repository. Download it and place it in `data/`, or attach it as a Kaggle input. Check the dataset's license before redistributing it.
- 211,043 rows: 161,043 attacks, 50,000 normal. Attack types: backdoor, ddos, dos, injection, password, ransomware, scanning, xss, mitm.
- The `type` column (attack category) is dropped because it leaks the label; IPs, ports and timestamps are dropped too.

## Model summary

| Item | Value |
|---|---|
| Type | Hybrid classical-quantum neural network |
| Qubits | 16 |
| Input features | All (about 127 after encoding), no feature selection |
| Classical front end | Linear(F -> 64), ReLU, Linear(64 -> 16), sigmoid |
| Encoding | Angle encoding RY(pi * x), angles learned by the front end |
| Circuit | 2 x [RX, RZ per qubit + CNOT chain] |
| Readout | <Z> on all 16 qubits -> Linear(16 -> 1) |
| Preprocessing | One-hot (<= 15 categories), frequency encoding, QuantileTransformer |
| Training | Adam, OneCycleLR, 8 epochs, batch 256 per GPU, gradient clipping 5.0 |
| Split | Stratified 80 / 10 / 10 |
| Baselines | None |

## Results (test split, threshold 0.219 tuned on validation)

| Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| 0.9932 | 0.9950 | 0.9961 | 0.9955 | 0.9988 | 0.9996 |

Full report in `results/final_results.txt`.

## Notes

- No classical-only baseline is included, so the contribution of the quantum layer is not isolated.
- No per-attack-type breakdown is reported.
- The CUDA-Q cross-check was skipped for the saved results (`pip install cuda-quantum-cu12` to enable it).
- Results are not comparable with V1-V3/V5: far more input features and an easier dataset.

## Usage

**Local**
```bash
pip install -r ../requirements.txt
# put the dataset in V4_HQNN_TONIOT/data/  (download links are in the Dataset section below)
cd V4_HQNN_TONIOT
python main.py
```

**Kaggle / notebook**: attach the dataset as an input (it is found under `/kaggle/input`) and run `python main.py`
or `import main; main.main()`.

- Data needed: `train_test_network.csv`.
- Hardware: GPU strongly recommended (16-qubit simulation); on CPU, set `MAX_SAMPLES` and reduce `N_QUBITS`.
- Settings: `src/config.py`. Results are written to `results/`.
