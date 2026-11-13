import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from src.config import CFG
from src.preprocess import CAT_COLS, ATTACK_FAMILY


def get_split(tr, te):
    """Validation = held-out part of the training file; test = the official test file. Everything is fitted on train rows."""
    seed, nq = CFG["SEED"], CFG["N_QUBITS"]
    seen = set(tr["label"])
    fam = lambda s: "Normal" if s == "normal" else ATTACK_FAMILY.get(s, "Other")
    fam_tr, fam_te = tr["label"].map(fam).to_numpy(), te["label"].map(fam).to_numpy()
    y_tr_all, y_te = tr["_y"].to_numpy(np.int64), te["_y"].to_numpy(np.int64)
    print(f"KDDTrain+: {len(tr):,} rows ({y_tr_all.mean():.3f} attack) | KDDTest+: {len(te):,} rows ({y_te.mean():.3f} attack)")
    print(f"Attack types in test never seen in training: {sorted(set(te['label']) - seen - {'normal'})}")

    # validation = held-out part of KDDTrain+ ; test = official KDDTest+
    strat = fam_tr if np.unique(fam_tr, return_counts=True)[1].min() >= 2 else y_tr_all
    itr, iva = train_test_split(np.arange(len(tr)), test_size=CFG["VAL_FRACTION"], stratify=strat, random_state=seed)

    cat_cols = [c for c in CAT_COLS if c in tr.columns]
    num_cols = [c for c in tr.columns if c not in cat_cols + ["label", "_y"]]
    num = lambda df: df[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32)
    N_tr, N_te = num(tr), num(te)

    # categorical -> smoothed attack rate (fitted on training rows only; unseen -> prior)
    prior = y_tr_all[itr].mean()
    e_tr, e_te, e_names = [], [], []
    for c in cat_cols:
        s_tr, s_te = tr[c].astype(str).str.strip(), te[c].astype(str).str.strip()
        cnt = s_tr.iloc[itr].value_counts()
        pos = s_tr.iloc[itr][y_tr_all[itr] == 1].value_counts().reindex(cnt.index).fillna(0)
        rate = (pos + 20 * prior) / (cnt + 20)
        e_tr.append(s_tr.map(rate).fillna(prior).to_numpy(np.float32))
        e_te.append(s_te.map(rate).fillna(prior).to_numpy(np.float32))
        e_names.append(c + "_rate")
    A_tr = np.hstack([N_tr, np.stack(e_tr, 1)]) if e_tr else N_tr
    A_te = np.hstack([N_te, np.stack(e_te, 1)]) if e_te else N_te
    names = num_cols + e_names

    # top-nq features by Random Forest importance (training rows only)
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=seed).fit(A_tr[itr], y_tr_all[itr])
    imp = pd.Series(rf.feature_importances_, index=names).sort_values(ascending=False)
    sel_names = imp.head(nq).index.tolist()
    sel = [names.index(c) for c in sel_names]
    print(f"\nTop {nq} features selected (qubit order):")
    for r, f in enumerate(sel_names):
        print(f"  qubit {r}: {f} ({imp[f]:.4f})")

    prep = lambda A: np.log1p(np.maximum(A[:, sel], 0))
    P_tr, P_te = prep(A_tr), prep(A_te)
    mn, mx = P_tr[itr].min(0), P_tr[itr].max(0)
    sc = lambda P: np.clip((P - mn) / np.where(mx > mn, mx - mn, 1.0), 0, 1).astype(np.float32)   # [0, 1]
    S_tr, S_te = sc(P_tr), sc(P_te)

    print(f"train {len(itr):,} | val {len(iva):,} | test {len(te):,}")
    return dict(Xtr=S_tr[itr], Xva=S_tr[iva], Xte=S_te, ytr=y_tr_all[itr], yva=y_tr_all[iva], yte=y_te,
                fam_va=fam_tr[iva], fam_te=fam_te, lab_te=te["label"].to_numpy(), seen=seen,
                imp=imp, sel_names=sel_names, counts=dict(train=y_tr_all, test=y_te))
