import os
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix,
                             precision_recall_curve)
from src.config import RESULTS_DIR
from src.hybrid_model import predict


def _metrics_at(y, p, t):
    yh = (p >= t).astype(int)
    return dict(accuracy=accuracy_score(y, yh), precision=precision_score(y, yh, zero_division=0),
                recall=recall_score(y, yh, zero_division=0), f1=f1_score(y, yh, zero_division=0),
                roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p),
                cm=confusion_matrix(y, yh), pred=yh)


def evaluate_model(net, Xva, yva, Xte, yte, cfg, device):
    print("\nEvaluating HQNN (threshold tuned on validation, metrics on the held-out TEST split)...")
    sig = lambda z: 1 / (1 + np.exp(-z))
    pva = sig(predict(net, Xva, device))
    pte = sig(predict(net, Xte, device))

    pr_, rc_, th_ = precision_recall_curve(yva, pva)
    f1s = 2 * pr_ * rc_ / (pr_ + rc_ + 1e-12)
    thr = float(th_[int(np.argmax(f1s[:-1]))]) if len(th_) else 0.5

    M05, MT = _metrics_at(yte, pte, 0.5), _metrics_at(yte, pte, thr)
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    lines = ["=" * 56, "  HQNN MODEL EVALUATION METRICS (TEST SET)", "=" * 56]
    for name, M in [("threshold 0.5", M05), (f"threshold {thr:.3f} (tuned on val)", MT)]:
        lines += [f"[{name}]"] + [f"  {k:<10}: {M[k]:.4f}" for k in keys]
        cm = M["cm"]
        lines += [f"  True Negatives (0):  {cm[0][0]:,}", f"  False Positives (1): {cm[0][1]:,}",
                  f"  False Negatives (0): {cm[1][0]:,}", f"  True Positives (1):  {cm[1][1]:,}", ""]
    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(RESULTS_DIR, "final_results.txt"), "w") as f:
        f.write(report)

    metrics = {k: MT[k] for k in keys}
    metrics["threshold"] = thr
    metrics["cm"] = MT["cm"]
    return yte, MT["pred"], pte, metrics
