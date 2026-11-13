import os
import math
import numpy as np
from sklearn.model_selection import train_test_split
from src.config import RESULTS_DIR
from src.preprocess import select_features


def split_dataset(arr, y, names, feat_names, cfg):
    """
    Stratified 80/10/10 train/val/test split of the FULL dataset, then (fitted on train only):
    feature selection -> signed log -> min-max -> L2 normalisation (required for amplitude encoding).
    """
    dim = 2 ** cfg["N_QUBITS"]
    idx = np.arange(len(y))
    tr, tmp = train_test_split(idx, test_size=0.2, stratify=y, random_state=cfg["SEED"])
    va, te = train_test_split(tmp, test_size=0.5, stratify=y[tmp], random_state=cfg["SEED"])

    sel, sel_names = select_features(arr, y, tr, feat_names, dim, cfg["SEED"])

    A = arr[:, sel]
    A = np.sign(A) * np.log1p(np.abs(A))                     # signed log (keeps negative values)
    mn, mx = A[tr].min(0), A[tr].max(0)
    den = np.where(mx - mn > 0, mx - mn, 1.0)
    A = np.clip((A - mn) / den, 0, 1).astype(np.float32)
    if A.shape[1] < dim:                                     # pad if fewer than `dim` features exist
        A = np.hstack([A, np.zeros((len(A), dim - A.shape[1]), np.float32)])

    nrm = np.linalg.norm(A, axis=1, keepdims=True)
    zero = nrm[:, 0] < 1e-8
    A = A / np.maximum(nrm, 1e-8)                            # sum of squares = 1.0 per row
    A[zero] = 1.0 / math.sqrt(dim)                           # all-zero rows -> valid uniform state
    np.savez(os.path.join(RESULTS_DIR, "preprocess.npz"), features=np.array(sel_names), mn=mn, mx=mx)

    print("\n--- Data Split & Normalized Successfully ---")
    print(f"train {len(tr):,} | val {len(va):,} | test {len(te):,} | features {A.shape[1]}")
    return A[tr], A[va], A[te], y[tr], y[va], y[te], names[te]
