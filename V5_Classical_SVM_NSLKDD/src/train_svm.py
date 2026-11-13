import time
import numpy as np
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import average_precision_score
from src.config import CFG


def fit_svc(X, y, C, gamma):
    t0 = time.time()
    m = SVC(kernel="rbf", C=C, gamma=gamma, cache_size=2000, random_state=CFG["SEED"]).fit(X, y)
    return m, time.time() - t0


def search_svc(D, gamma0):
    """Grid over C and gamma on a subsample; scored by PR-AUC on the FULL validation split."""
    rs = np.random.RandomState(CFG["SEED"])
    idx = rs.choice(len(D["Xtr"]), min(CFG["SEARCH_ROWS"], len(D["Xtr"])), replace=False)
    grid = {}
    print(f"\nHyper-parameter search on {len(idx):,} training rows (val PR-AUC):")
    for C in CFG["C_GRID"]:
        for gm in CFG["GAMMA_MULTS"]:
            m, t = fit_svc(D["Xtr"][idx], D["ytr"][idx], C, gamma0 * gm)
            grid[(C, gm)] = average_precision_score(D["yva"], m.decision_function(D["Xva"]))
            print(f"  C={C:<6g} gamma={gm:g} x scale -> {grid[(C, gm)]:.4f}  ({t:.1f}s)")
    best = max(grid, key=grid.get)
    print(f"Best: C={best[0]:g}, gamma={best[1]:g} x scale (val PR-AUC {grid[best]:.4f})")
    return grid, best


def train_models(D):
    """Exact RBF SVMs on ALL training rows (original settings + tuned) and a linear reference.
    Returns ({name: (estimator, fit seconds)}, search grid)."""
    Xtr, ytr = D["Xtr"], D["ytr"]
    gamma0 = 1.0 / (Xtr.shape[1] * Xtr.var())                # = sklearn's gamma='scale'
    print(f"\ngamma='scale' = {gamma0:.4f} | training rows: {len(Xtr):,}")

    fitted = {}                                              # name -> (estimator, fit seconds)
    m, t = fit_svc(Xtr, ytr, 1.0, gamma0)
    n0 = "RBF SVM original settings (C=1, gamma=scale)"
    fitted[n0] = (m, t)
    print(f"\n{n0}: fit on ALL {len(Xtr):,} rows in {t:.1f}s | support vectors: {int(m.n_support_.sum()):,}")
    grid, best = ({}, None)
    if CFG["SEARCH"]:
        grid, best = search_svc(D, gamma0)
        if best != (1.0, 1.0):
            m, t = fit_svc(Xtr, ytr, best[0], gamma0 * best[1])
            n1 = f"RBF SVM tuned (C={best[0]:g}, gamma={best[1]:g}x scale)"
            fitted[n1] = (m, t)
            print(f"\n{n1}: fit on ALL rows in {t:.1f}s | support vectors: {int(m.n_support_.sum()):,}")
    if CFG["RUN_LINEAR"]:
        t0 = time.time()
        m = LinearSVC(C=1.0, dual=False, max_iter=5000, random_state=CFG["SEED"]).fit(Xtr, ytr)
        fitted["Linear SVM (reference)"] = (m, time.time() - t0)
    return fitted, grid
