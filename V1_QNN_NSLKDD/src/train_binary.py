import copy
import math
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from src.config import CFG
from src.quantum_classifier import QNN
from src.baselines import Logistic, MLP
from src.objective import scores


def fit(model, name, D, device):
    g = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
    Xtr, Xva, ytr = g(D["Xtr"]), g(D["Xva"]), g(D["ytr"].astype(np.float32))
    n_pos = max(int(D["ytr"].sum()), 1)
    w_pos = math.sqrt((len(ytr) - n_pos) / n_pos)             # square-root smoothed class weight
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
            loss = bce(model(Xtr[ix]), ytr[ix])
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
        if ep == 1 or ep % 10 == 0 or ep == CFG["EPOCHS"]:
            print(f"  epoch {ep:02d}/{CFG['EPOCHS']} | loss {hist['train_loss'][-1]:.4f} | "
                  f"val PR-AUC {ap:.4f} | {time.time() - t0:.0f}s")
    model.load_state_dict(best_state)
    return model, hist, best_ap


def fit_best(make, name, D, device, restarts):
    best = None
    for r in range(restarts):
        torch.manual_seed(CFG["SEED"] + r)
        label = f"{name} [restart {r + 1}/{restarts}]" if restarts > 1 else name
        m, h, ap = fit(make(), label, D, device)
        if best is None or ap > best[2]:
            best = (m, h, ap)
    return best


def train_all(D, device):
    """Improved QNN (first), original-style QNN, classical baselines. Returns {name: (model, history)}."""
    nq, nl, d = CFG["N_QUBITS"], CFG["N_LAYERS"], D["Xtr"].shape[1]
    models = {}
    q_name = f"QNN improved ({nl} layers, all-qubit readout)"
    m, h, _ = fit_best(lambda: QNN(nq, nl, "all"), q_name, D, device, CFG["N_RESTARTS"])
    models[q_name] = (m, h)
    if CFG["RUN_ORIGINAL_STYLE"]:
        o_name = "QNN original-style (2 layers, mean Z0-3)"
        m, h, _ = fit_best(lambda: QNN(nq, 2, "mean4"), o_name, D, device, CFG["N_RESTARTS"])
        models[o_name] = (m, h)
    if CFG["RUN_BASELINES"]:
        for name, make in [("Logistic regression", lambda: Logistic(d)), ("MLP 64-64", lambda: MLP(d))]:
            m, h, _ = fit_best(make, name, D, device, 1)
            models[name] = (m, h)
    return models
