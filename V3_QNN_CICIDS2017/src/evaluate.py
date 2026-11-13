import os
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix,
                             precision_recall_curve)
from src.config import RESULTS_DIR
from src.quantum_model import predict


def _metrics_at(y, p, t):
    yh = (p >= t).astype(int)
    return dict(accuracy=accuracy_score(y, yh), precision=precision_score(y, yh, zero_division=0),
                recall=recall_score(y, yh, zero_division=0), f1=f1_score(y, yh, zero_division=0),
                roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p),
                cm=confusion_matrix(y, yh), pred=yh)


def evaluate_model(model, Xva, yva, Xte, yte, names_te, cfg, device):
    print("\nEvaluating QNN (threshold tuned on validation, metrics on the held-out TEST split)...")
    to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
    sig = lambda z: 1 / (1 + np.exp(-z))
    pva = sig(predict(model, to(Xva)).cpu().numpy().ravel())
    pte = sig(predict(model, to(Xte)).cpu().numpy().ravel())

    # --- DYNAMIC THRESHOLD TUNING (on validation): maximise F1 ---
    pr_, rc_, th_ = precision_recall_curve(yva, pva)
    f1s = 2 * pr_ * rc_ / (pr_ + rc_ + 1e-10)
    best_idx = int(np.argmax(f1s[:-1]))
    thr = float(th_[best_idx]) if len(th_) else 0.5
    print(f"[Dynamic Tuning] Optimal Probability Threshold found: {thr:.4f}")

    M05, MT = _metrics_at(yte, pte, 0.5), _metrics_at(yte, pte, thr)
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    lines = ["=" * 56, "  QNN (V3, CICIDS2017) TEST-SET METRICS", "=" * 56]
    for name, M, t in [("threshold 0.5", M05, 0.5), (f"threshold {thr:.4f} (tuned on val)", MT, thr)]:
        lines += [f"[{name}]"] + [f"  {k:<10}: {M[k]:.4f}" for k in keys]
        cm = M["cm"]
        lines += [f"  True Negatives (0):  {cm[0][0]:,}", f"  False Positives (1): {cm[0][1]:,}",
                  f"  False Negatives (0): {cm[1][0]:,}", f"  True Positives (1):  {cm[1][1]:,}", ""]
    lines += [f"Per-class results on test (tuned threshold {thr:.4f}):"]
    for nm in sorted(set(names_te)):
        m = names_te == nm
        if nm == "BENIGN":
            lines.append(f"  {nm:<28} n={m.sum():>8,}  correctly kept benign: {(MT['pred'][m] == 0).mean():.4f}")
        else:
            lines.append(f"  {nm:<28} n={m.sum():>8,}  detected:             {(MT['pred'][m] == 1).mean():.4f}")
    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(RESULTS_DIR, "final_results.txt"), "w") as f:
        f.write(report)
    np.savez(os.path.join(RESULTS_DIR, "threshold.npz"), thr=thr)

    metrics = {k: MT[k] for k in keys}
    metrics["threshold"] = thr
    metrics["cm"] = MT["cm"]
    return MT["pred"], pte, metrics
