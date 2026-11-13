import os
import json
import math
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from src.config import RESULTS_DIR
from src.train_qsvm import scores


def best_f1_threshold(y, s):
    p, r, t = precision_recall_curve(y, s)
    f1 = 2 * p * r / (p + r + 1e-12)
    return float(t[int(np.argmax(f1[:-1]))])


def metrics_at(y, s, t):
    y = y.astype(bool)
    pred = s >= t
    tp = float((pred & y).sum()); fp = float((pred & ~y).sum())
    fn = float((~pred & y).sum()); tn = float((~pred & ~y).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return dict(threshold=float(t), accuracy=(tp + tn) / len(y), precision=prec, recall=rec,
                f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                mcc=(tp * tn - fp * fn) / den if den > 0 else 0.0,
                roc_auc=float(roc_auc_score(y, s)), pr_auc=float(average_precision_score(y, s)),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp), pred=pred)


def evaluate_all(models, D, device):
    """Threshold tuned on validation (max F1); metrics on the held-out TEST split. Returns (runs, report)."""
    g = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
    Xva_g, Xte_g = g(D["Xva"]), g(D["Xte"])
    runs = {}
    for n, (m, hist) in models.items():
        sv, st = scores(m, Xva_g), scores(m, Xte_g)
        thr = best_f1_threshold(D["yva"], sv)
        runs[n] = dict(hist=hist, test_scores=st, default=metrics_at(D["yte"], st, 0.0),
                       tuned=metrics_at(D["yte"], st, thr))

    cols = ["accuracy", "precision", "recall", "f1", "mcc", "roc_auc", "pr_auc"]

    def table(key):
        out = [f"{'Model':<22}" + "".join(f"{c:>10}" for c in cols)]
        for n, r in runs.items():
            out.append(f"{n:<22}" + "".join(f"{r[key][c]:>10.4f}" for c in cols))
        return "\n".join(out)

    lines = ["=" * 90, "TEST RESULTS - threshold tuned on validation (max F1)", "=" * 90, table("tuned"), "",
             "TEST RESULTS - default threshold (score >= 0)", table("default"), "",
             "Detection rate per class on TEST (tuned threshold; Normal = correctly kept normal):"]
    lines.append(f"{'Class':<18}{'n':>10}" + "".join(f"{n[:18]:>20}" for n in runs))
    for c in sorted(set(D["names_te"])):
        mask = D["names_te"] == c
        row = f"{c:<18}{int(mask.sum()):>10,}"
        for r in runs.values():
            p = r["tuned"]["pred"][mask]
            row += f"{(1 - p.mean() if c == 'Normal' else p.mean()):>20.4f}"
        lines.append(row)
    report = "\n".join(lines)
    print("\n" + report)

    with open(os.path.join(RESULTS_DIR, "final_results.txt"), "w") as f:
        f.write(report + "\n")
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump({n: {"tuned": {k: v for k, v in r["tuned"].items() if k != "pred"},
                       "default": {k: v for k, v in r["default"].items() if k != "pred"},
                       "history": r["hist"]} for n, r in runs.items()}, f, indent=2)
    return runs, report
