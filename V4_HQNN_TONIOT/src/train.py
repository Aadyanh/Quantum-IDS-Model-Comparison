import os
import time
import copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from src.config import RESULTS_DIR
from src.hybrid_model import predict


def train_hybrid_model(model, net, Xtr, ytr, Xva, yva, cfg, device):
    """Adam + exact gradients on the FULL training split. Keeps the best epoch (validation loss)."""
    print("\nStarting Hybrid QNN training using PyTorch Adam Optimizer...")
    n_gpu = torch.cuda.device_count()
    use_dp = isinstance(net, nn.DataParallel)
    bs = min(cfg["BATCH_PER_GPU"] * (n_gpu if use_dp else 1), len(Xtr))
    steps = len(Xtr) // bs
    total_steps = steps * cfg["EPOCHS"]

    Xtr_g = torch.from_numpy(np.ascontiguousarray(Xtr)).to(device)
    ytr_g = torch.from_numpy(ytr.astype(np.float32)).view(-1, 1).to(device)
    yva_t = torch.from_numpy(yva.astype(np.float32)).view(-1, 1)

    n_pos = int(ytr.sum())
    pos_weight = (len(ytr) - n_pos) / max(n_pos, 1)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    crit_cpu = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    opt = torch.optim.Adam(model.parameters(), lr=cfg["LR"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["LR"], total_steps=total_steps, pct_start=0.15)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,} | batch {bs} | "
          f"{steps} steps/epoch | epochs {cfg['EPOCHS']} | pos_weight {pos_weight:.2f}")

    hist = dict(train=[], val=[], val_auc=[])
    best = dict(loss=float("inf"), state=None, epoch=0)
    t0, stop = time.time(), False
    log_every = max(1, steps // 5)

    for ep in range(1, cfg["EPOCHS"] + 1):
        net.train()
        perm = torch.randperm(len(Xtr_g), device=device)
        run = torch.zeros((), device=device)
        done = 0
        for it in range(steps):
            ix = perm[it * bs:(it + 1) * bs]
            opt.zero_grad(set_to_none=True)
            loss = crit(net(Xtr_g[ix]), ytr_g[ix])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            run += loss.detach()
            done += 1
            if (it + 1) % log_every == 0:
                print(f"  Epoch [{ep}/{cfg['EPOCHS']}] | Batch [{it + 1}/{steps}] | "
                      f"Loss: {loss.item():.4f} | {(time.time() - t0) / 60:.1f} min elapsed")
            if time.time() - t0 > cfg["TIME_BUDGET_MIN"] * 60:
                stop = True
                break
        tr_loss = (run / max(done, 1)).item()
        lv = predict(net, Xva, device)
        val_loss = crit_cpu(torch.from_numpy(lv).view(-1, 1), yva_t).item()
        val_auc = roc_auc_score(yva, lv)
        hist["train"].append(tr_loss)
        hist["val"].append(val_loss)
        hist["val_auc"].append(val_auc)
        print(f"=== Epoch [{ep}/{cfg['EPOCHS']}] Complete | train {tr_loss:.4f} | "
              f"val {val_loss:.4f} | val AUC {val_auc:.4f} ===")
        if val_loss < best["loss"]:
            best.update(loss=val_loss, state=copy.deepcopy(model.state_dict()), epoch=ep)
        if stop:
            print("Time budget reached - stopping early.")
            break

    model.load_state_dict(best["state"])
    model_path = os.path.join(RESULTS_DIR, "hybrid_qnn_model.pth")
    torch.save(model.state_dict(), model_path)
    np.savetxt(os.path.join(RESULTS_DIR, "training_loss.txt"), np.array(hist["train"]))
    print("\n--- Training Complete ---")
    print(f"Best epoch: {best['epoch']} (val loss {best['loss']:.4f}) | model saved to {model_path}")
    return model, hist
