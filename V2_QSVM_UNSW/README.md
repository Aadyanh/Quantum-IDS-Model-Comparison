# V2 QSVM UNSW-NB15 Pipeline (GPU version)

A **Quantum Support Vector Machine (QSVM)** for network intrusion detection on **UNSW-NB15**.
6 features -> 6 qubits (angle encoding `RY(pi * scale * x)`), fidelity quantum kernel `K(x,x') = |<psi(x)|psi(x')>|^2`.

The 6-qubit angle-encoded state is a product state, so `K(x,x') = (psi(x) . psi(x'))^2` and the SVM can be trained in its
**primal form** on the GPU: `f(x) = psi(x)^T W psi(x) + b` (W symmetric 64x64, hinge loss). This is the same kernel SVM,
but it never builds an n x n Gram matrix, so it trains on the **whole dataset** (millions of rows).
CUDA-Q is used afterwards to check that this kernel equals the original CUDA-Q statevector kernel.

## Project Structure

```text
V2_QSVM_UNSWNB15/
├── main.py                 # runs the whole pipeline
├── data/                   # UNSW-NB15_1..4.csv (not tracked by git)
├── results/                # model weights, metrics, plots
└── src/
    ├── config.py           # all settings in one place
    ├── preprocess.py       # loading, cleaning, de-duplication
    ├── split_data.py       # stratified 70/15/15 split, target encoding, RF feature selection, scaling
    ├── quantum_kernel.py   # GPU quantum-kernel SVM (+ CUDA-Q check)
    ├── baselines.py        # classical linear SVM and MLP on the same features
    ├── train_qsvm.py       # training, encoding-scale search on validation
    ├── evaluate.py         # threshold tuned on validation, metrics on the held-out test split
    ├── visualize_data.py   # ROC / PR / confusion matrices / metrics / feature plots
    └── cudaq_verify.py     # standalone CUDA-Q kernel check
```

## What the pipeline does

1. **Load** all four CSVs, remove duplicate flows, drop identifiers and timestamps (they leak the label).
2. **Split first**: stratified 70 / 15 / 15 train / validation / test.
3. **Fit on train only**: categorical columns -> attack rate, Random Forest picks the top 6 features, log + min-max scaling to [0, 1].
4. **Train** the QSVM with Adam on the full training split; the encoding scale (0.5 / 1.0 / 1.5) is chosen on validation.
   Class imbalance is handled with a square-root smoothed class weight.
5. **Compare** with a classical linear SVM and a small MLP on the same 6 features.
6. **Evaluate**: probability/score threshold maximising F1 on validation; accuracy, precision, recall, F1, MCC, ROC-AUC, PR-AUC,
   confusion matrices and a per-attack-type detection table on the TEST split.

## Usage

**Local**
```bash
pip install -r ../requirements.txt
# put the dataset in V2_QSVM_UNSW/data/  (download links are in the Dataset section below)
cd V2_QSVM_UNSW
python main.py
```

**Kaggle / notebook**: attach the dataset as an input (it is found under `/kaggle/input`) and run `python main.py`
or `import main; main.main()`.

- Data needed: UNSW-NB15_1..4.csv.
- Hardware: GPU recommended; on CPU, use `MAX_SAMPLES` in `src/config.py` (e.g. 300_000) for a quick run.
- Settings: `src/config.py`. Results are written to `results/`.

## Dataset

- **UNSW-NB15**, files `UNSW-NB15_1.csv` ... `UNSW-NB15_4.csv`, official source: https://research.unsw.edu.au/projects/unsw-nb15-dataset
- Kaggle mirror: https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15
- Not included in this repository. Download it and place the files in `data/` (or attach it as a Kaggle input).
- Task: binary (normal vs attack), about 4% attacks after de-duplication, 9 attack categories.

## Model summary

| Item | Value |
|---|---|
| Type | Quantum kernel SVM (fidelity kernel), trained in primal form |
| Qubits | 6 (one feature per qubit) |
| Features | Top 6 by Random Forest importance |
| Encoding | Angle encoding, RY(pi * s * x), s chosen from {0.5, 1.0, 1.5} (best 1.5) |
| Entanglement | None (product state) |
| Kernel | K(x, x') = abs(<psi(x)|psi(x')>)^2 |
| Loss / optimizer | Weighted hinge + L2, Adam, 30 epochs, batch 8192 |
| Split | Stratified 70 / 15 / 15 |
| Baselines | Linear SVM, MLP 64-64 |

## Results (test split, threshold tuned on validation)

| Model | Accuracy | Precision | Recall | F1 | PR-AUC |
|---|---|---|---|---|---|
| QSVM (scale 1.5) | 0.9836 | 0.7282 | 0.9927 | 0.8401 | 0.8743 |
| Linear SVM | 0.9826 | 0.7179 | 0.9834 | 0.8299 | 0.8298 |
| MLP 64-64 | 0.9829 | 0.7173 | 0.9985 | 0.8349 | 0.8399 |

Full tables are in `results/final_results.txt` and `results/results.json`.

## Notes

- Without threshold tuning (score >= 0) QSVM recall is low (0.389), so the tuned threshold matters.
- The product-state kernel can be computed classically; no quantum advantage is claimed.
- `src/cudaq_verify.py` can cross-check the kernel against CUDA-Q; this check was not run for the saved results.
- See the root README for the comparison of all five versions.
