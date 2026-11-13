import copy
import math
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from src.config import CFG
from src.quantum_kernel import QSVM
from src.baselines import LinearSVM, MLP


@torch.no_grad()
def scores(model, X, chunk=65536):
    model.eval()
    return torch.cat([model(X[i:i + chunk]) for i in range(0, len(X), chunk)]).cpu().numpy()


def fit(model, name, kind, D, device):
    """kind='hinge' -> SVM objective (weighted hinge + L2); kind='bce' -> logistic loss."""
    g = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
    Xtr, Xva, ytr = g(D["Xtr"]), g(D["Xva"]), g(D["ytr"].astype(np.float32))
    n_pos = max(int(D["ytr"].sum()), 1)
    w_pos = math.sqrt((len(ytr) - n_pos) / n_pos)         # square-root smoothed class weight
    bs = min(CFG["BATCH"], len(Xtr))
    steps = max(1, len(Xtr) // bs)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n=== {name} | params {n_params:,} | {len(Xtr):,} train rows | {steps} steps/epoch ===")

    opt = torch.optim.Adam(model.parameters(), lr=CFG["LR"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=CFG["LR"], total_steps=steps * CFG["EPOCHS"], pct_start=0.1)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(w_pos, device=device))

    hist = {"train_loss": [], "val_ap": []}
    best_ap, best_state, t0 = -1.0, None, time.time()
    for ep in range(1, CFG["EPOCHS"] + 1):
        model.train()
        perm = torch.randperm(len(Xtr), device=device)
        run = torch.zeros((), device=device)
        for it in range(steps):
            ix = perm[it * bs:(it + 1) * bs]
            xb, yb = Xtr[ix], ytr[ix]
            s = model(xb)
            if kind == "hinge":
                w = torch.where(yb > 0.5, torch.full_like(yb, w_pos), torch.ones_like(yb))
                loss = (w * torch.relu(1 - (2 * yb - 1) * s)).mean() + 0.5 * CFG["L2"] * model.l2()
            else:
                loss = bce(s, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run += loss.detach()
        ap = average_precision_score(D["yva"], scores(model, Xva))
        hist["train_loss"].append((run / steps).item())
        hist["val_ap"].append(ap)
        if ap > best_ap:
            best_ap, best_state = ap, copy.deepcopy(model.state_dict())
        if ep == 1 or ep % 5 == 0 or ep == CFG["EPOCHS"]:
            print(f"  epoch {ep:02d}/{CFG['EPOCHS']} | loss {hist['train_loss'][-1]:.4f} | "
                  f"val PR-AUC {ap:.4f} | {time.time() - t0:.0f}s")
    model.load_state_dict(best_state)
    return model, hist, best_ap


def train_all(D, device):
    """QSVM (encoding scale chosen on validation) + classical baselines. Returns ({name: (model, history)}, search)."""
    d = D["Xtr"].shape[1]
    search, best = [], None
    for s in CFG["ENC_SCALES"]:
        m, hist, ap = fit(QSVM(s, CFG["N_QUBITS"]), f"QSVM (scale {s})", "hinge", D, device)
        search.append((s, ap))
        if best is None or ap > best[2]:
            best = (s, m, ap, hist)
    print("\nScale search (val PR-AUC):", {s: round(a, 4) for s, a in search}, "-> best scale", best[0])
    models = {f"QSVM (scale {best[0]})": (best[1], best[3])}
    if CFG["RUN_BASELINES"]:
        for name, mdl, kind in [("Linear SVM", LinearSVM(d), "hinge"), ("MLP 64-64", MLP(d), "bce")]:
            m, hist, _ = fit(mdl, name, kind, D, device)
            models[name] = (m, hist)
    return models, search
