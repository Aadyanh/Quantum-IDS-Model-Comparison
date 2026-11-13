# V5 Classical RBF SVM Baseline - NSL-KDD (full data)

The classical baseline for the quantum NSL-KDD model: an **RBF-kernel SVM** (sklearn / libsvm) on the same 8 features,
the same validation split and the same official test set (`KDDTest+`), so results are directly comparable.
It is trained on **all** training rows (the original used 1,000); with 8 features this takes about 30 s on CPU, so no GPU is needed.

## Project Structure

```text
V5_Classical_SVM_NSLKDD/
├── main.py                 # runs the whole pipeline
├── data/                   # KDDTrain / KDDTest (.parquet, .txt or .csv) - not tracked by git
├── results/                # metrics and plots
└── src/
    ├── config.py           # all settings in one place
    ├── preprocess.py       # reads parquet/csv/txt, detects columns and the label column
    ├── split_data.py       # train/validation split, target encoding, RF feature selection, scaling
    ├── train_svm.py        # exact RBF SVMs (original settings + tuned C/gamma) and a linear reference
    ├── evaluate.py         # threshold tuned on validation, metrics, per-class detection, timings
    └── visualize_data.py   # plots
```

## What the pipeline does

1. **Load** KDDTrain and KDDTest (the official split; the test file contains 17 attack types never seen in training).
2. **Split**: KDDTrain -> 85% train / 15% validation (stratified by attack family). KDDTest is only used for testing.
3. **Fit on train only**: categorical columns -> attack rate, Random Forest picks the top 8 features, log + min-max scaling to [0, 1].
4. **Train** the RBF SVM with the original settings (C=1, gamma='scale') on all rows, search C and gamma on validation
   (on a subsample), refit the best setting on all rows, and train a linear SVM as a reference.
5. **Evaluate**: threshold maximising F1 on validation; accuracy, precision, recall, F1, MCC, ROC-AUC, PR-AUC on validation
   and on the official test set, detection rate per attack family, seen vs. novel attacks, fit/inference times.
   Scores are the SVM decision function (no `probability=True`, which would slow training ~5x).

## Usage

**Local**
```bash
pip install -r ../requirements.txt
# put the dataset in V5_Classical_SVM_NSLKDD/data/  (download links are in the Dataset section below)
cd V5_Classical_SVM_NSLKDD
python main.py
```

**Kaggle / notebook**: attach the dataset as an input (it is found under `/kaggle/input`) and run `python main.py`
or `import main; main.main()`.

- Data needed: NSL-KDD files (KDDTrain+, KDDTest+).
- Hardware: CPU only, no GPU needed (about 30 s per SVM fit).
- Settings: `src/config.py`. Results are written to `results/`.

## Dataset

- **NSL-KDD** (KDDTrain+ / KDDTest+), official source: https://www.unb.ca/cic/datasets/nsl.html
- Kaggle mirror: https://www.kaggle.com/datasets/hassan06/nslkdd
- Not included in this repository. Download it and place the files in `data/` (or attach it as a Kaggle input).

## Model summary

| Item | Value |
|---|---|
| Type | Classical RBF-kernel SVM (scikit-learn) |
| Features | Same 8 features as V1 |
| Search | C in {1, 10, 100} x gamma multiples {0.5, 1, 2, 4}, on a 30,000-row subsample |
| Best | C=100, gamma=2 x 'scale' |
| Also trained | RBF SVM with C=1 gamma='scale', Linear SVM |

## Results (KDDTest+, threshold tuned on validation)

| Model | Accuracy | F1 | ROC-AUC | Fit time |
|---|---|---|---|---|
| RBF original (C=1) | 0.7958 | 0.7880 | 0.9325 | 26.3 s |
| RBF tuned (C=100, 2x gamma) | 0.8442 | 0.8454 | 0.9508 | 24.5 s |
| Linear SVM | 0.7728 | 0.7720 | 0.9375 | 0.1 s |

Full tables are in `results/final_results.txt` and `results/results.json`.

## Notes

- Classical baseline for V1: same data, split, preprocessing and features.
- Runs on CPU only.
- See the root README for the comparison of all five versions.
