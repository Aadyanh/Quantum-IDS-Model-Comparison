import os
import time
import copy
import math
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from src.config import RESULTS_DIR
from src.quantum_model import QNN, predict


def train_qnn(Xtr, ytr, Xva, yva, cfg, device):
    """Adam + exact gradients on the full training split, with random restarts. Returns (best model, history)."""
    g = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
    Xtr_g, Xva_g = g(Xtr), g(Xva)
    ytr_g = g(ytr.astype(np.float32)).view(-1, 1)
    yva_g = g(yva.astype(np.float32)).view(-1, 1)

    # SQUARE ROOT SMOOTHING of the class imbalance (as in the original V3)
    n_pos = int(ytr.sum())
    n_neg = len(ytr) - n_pos
    pos_weight = math.sqrt(n_neg / max(n_pos, 1))
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    bs = min(cfg["BATCH_SIZE"], len(Xtr))
    steps = len(Xtr) // bs
    print(f"\nStarting QNN training with Adam | batch {bs} | {steps} steps/epoch | pos_weight {pos_weight:.2f}")

    t0 = time.time()
    budget = cfg["TIME_BUDGET_MIN"] * 60
    best = dict(loss=float("inf"), state=None, hist=None, restart=-1, epoch=-1)

    for r in range(cfg["N_RESTARTS"]):
        if time.time() - t0 > budget:
            break
        torch.manual_seed(cfg["SEED"] + r)
        model = QNN(cfg["N_QUBITS"], cfg["N_LAYERS"]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg["LR"])
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["LR"],
                                                    total_steps=steps * cfg["EPOCHS"], pct_start=0.15)
        hist = dict(train=[], val=[], val_auc=[])
        r_best = float("inf")
        print(f"\n--- Restart {r + 1}/{cfg['N_RESTARTS']} ---")
        for ep in range(1, cfg["EPOCHS"] + 1):
            model.train()
            perm = torch.randperm(len(Xtr_g), device=device)
            run = torch.zeros((), device=device)
            for it in range(steps):
                ix = perm[it * bs:(it + 1) * bs]
                opt.zero_grad(set_to_none=True)
                loss = crit(model(Xtr_g[ix]), ytr_g[ix])
                loss.backward()
                opt.step()
                sched.step()
                run += loss.detach()
            tl = (run / steps).item()
            lv = predict(model, Xva_g)
            vl = crit(lv, yva_g).item()
            va_auc = roc_auc_score(yva, lv.cpu().numpy().ravel())
            hist["train"].append(tl)
            hist["val"].append(vl)
            hist["val_auc"].append(va_auc)
            r_best = min(r_best, vl)
            if vl < best["loss"]:
                best.update(loss=vl, state=copy.deepcopy(model.state_dict()),
                            hist=copy.deepcopy(hist), restart=r, epoch=ep)
            if ep == 1 or ep % 5 == 0 or ep == cfg["EPOCHS"]:
                print(f"  ep {ep:02d}/{cfg['EPOCHS']} | train {tl:.4f} | val {vl:.4f} | "
                      f"val AUC {va_auc:.4f} | {(time.time() - t0) / 60:.1f} min")
            if time.time() - t0 > budget:
                print("Time budget reached - stopping.")
                break
        print(f"  restart {r + 1} best val loss: {r_best:.4f}")

    model = QNN(cfg["N_QUBITS"], cfg["N_LAYERS"]).to(device)
    model.load_state_dict(best["state"])
    model.eval()
    print(f"\nBest: restart {best['restart'] + 1}, epoch {best['epoch']} (val loss {best['loss']:.4f})")

    torch.save(model.state_dict(), os.path.join(RESULTS_DIR, "qnn_model.pth"))
    np.save(os.path.join(RESULTS_DIR, "optimized_theta.npy"), model.thetas.detach().cpu().numpy())
    np.savetxt(os.path.join(RESULTS_DIR, "training_loss.txt"), np.array(best["hist"]["train"]))
    print("\n--- Training Complete ---")
    return model, best["hist"]
