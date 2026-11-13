import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from src.config import CFG, RESULTS_DIR


def split_dataset(num, num_cols, cats, y, names):
    """Split FIRST; everything below is fitted on the training rows only."""
    seed, nf = CFG["SEED"], CFG["N_QUBITS"]
    idx = np.arange(len(y))
    tr, tmp = train_test_split(idx, test_size=0.30, stratify=y, random_state=seed)
    va, te = train_test_split(tmp, test_size=0.50, stratify=y[tmp], random_state=seed)

    # categorical -> smoothed attack rate (target encoding, train rows only)
    prior = y[tr].mean()
    enc, enc_names = [], []
    for c, v in cats.items():
        codes, uniq = pd.factorize(v)
        n = np.bincount(codes[tr], minlength=len(uniq)).astype(np.float64)
        pos = np.bincount(codes[tr], weights=y[tr], minlength=len(uniq))
        enc.append((((pos + 20 * prior) / (n + 20))[codes]).astype(np.float32))
        enc_names.append(c + "_rate")
    arr = np.hstack([num, np.stack(enc, 1)])
    feat_names = num_cols + enc_names

    # top-nf features by Random Forest importance (train rows only)
    samp = np.random.RandomState(seed).choice(tr, size=min(100_000, len(tr)), replace=False)
    rf = RandomForestClassifier(n_estimators=50, min_samples_leaf=2, n_jobs=-1, random_state=seed)
    rf.fit(arr[samp], y[samp])
    imp = pd.Series(rf.feature_importances_, index=feat_names).sort_values(ascending=False)
    sel_names = imp.head(nf).index.tolist()
    print(f"\nTop {nf} features selected:")
    for r, f in enumerate(sel_names, 1):
        print(f"  {r}. {f} ({imp[f]:.4f})")

    A = np.log1p(np.maximum(arr[:, [feat_names.index(c) for c in sel_names]], 0))
    mn, mx = A[tr].min(0), A[tr].max(0)
    A = np.clip((A - mn) / np.where(mx > mn, mx - mn, 1.0), 0, 1).astype(np.float32)   # [0, 1]

    np.savez(os.path.join(RESULTS_DIR, "preprocess.npz"), features=np.array(sel_names), mn=mn, mx=mx)
    print(f"train {len(tr):,} | val {len(va):,} | test {len(te):,}")
    return dict(Xtr=A[tr], Xva=A[va], Xte=A[te], ytr=y[tr], yva=y[va], yte=y[te],
                names_te=names[te], imp=imp, sel_names=sel_names)
