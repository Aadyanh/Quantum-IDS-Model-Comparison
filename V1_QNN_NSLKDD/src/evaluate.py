import math
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from src.objective import scores, feature_influence


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
    """Threshold tuned on validation (max F1); metrics on validation and on the OFFICIAL test set. Returns (runs, report)."""
    # ---- evaluate: threshold tuned on validation; metrics on validation and on the OFFICIAL test set ----
    g = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)
    Xva_g, Xte_g = g(D["Xva"]), g(D["Xte"])
    in_seen = np.array([lab in D["seen"] for lab in D["lab_te"]])
    fam_groups = ["Normal", "DoS", "Probe", "R2L", "U2R"] + (["Other"] if (D["fam_te"] == "Other").any() else [])
    runs = {}
    for n, (m, hist) in models.items():
        sv, st = scores(m, Xva_g), scores(m, Xte_g)
        thr = best_f1_threshold(D["yva"], sv)
        r = dict(hist=hist, test_scores=st, thr=thr, val=metrics_at(D["yva"], sv, thr),
                 test=metrics_at(D["yte"], st, thr), test_default=metrics_at(D["yte"], st, 0.0),
                 influence=feature_influence(m, Xva_g))
        pred = r["test"]["pred"]
        rate = lambda mask: float(pred[mask].mean()) if mask.sum() else float("nan")
        det = {"Normal": 1 - rate(D["fam_te"] == "Normal")}
        for f in fam_groups[1:]:
            det[f] = rate(D["fam_te"] == f)
        det["Seen attacks"] = rate((D["yte"] == 1) & in_seen)
        det["Novel attacks"] = rate((D["yte"] == 1) & ~in_seen)
        r["det"] = det
        runs[n] = r

    cols = ["accuracy", "precision", "recall", "f1", "mcc", "roc_auc", "pr_auc"]

    def table(key):
        out = [f"{'Model':<44}" + "".join(f"{c:>10}" for c in cols)]
        for n, r in runs.items():
            out.append(f"{n:<44}" + "".join(f"{r[key][c]:>10.4f}" for c in cols))
        return "\n".join(out)

    groups = fam_groups + ["Seen attacks", "Novel attacks"]
    n_te = {f: int((D["fam_te"] == f).sum()) for f in fam_groups}
    n_te["Seen attacks"] = int(((D["yte"] == 1) & in_seen).sum())
    n_te["Novel attacks"] = int(((D["yte"] == 1) & ~in_seen).sum())
    lines = ["=" * 100, "OFFICIAL TEST SET (KDDTest+) - threshold tuned on validation (max F1)", "=" * 100, table("test"), "",
             "OFFICIAL TEST SET - default threshold (logit >= 0)", table("test_default"), "",
             "VALIDATION (held-out 15% of KDDTrain+, same distribution as training) - tuned threshold", table("val"), "",
             "Detection rate per class on the OFFICIAL TEST SET (Normal = correctly kept normal):",
             f"{'Class':<16}{'n':>8}" + "".join(f"{n[:26]:>28}" for n in runs)]
    for gname in groups:
        fmt = lambda v: f"{v:>28.4f}" if v == v else f"{'n/a':>28}"
        lines.append(f"{gname:<16}{n_te[gname]:>8,}" + "".join(fmt(r["det"][gname]) for r in runs.values()))
    lines += ["", "Share (%) of each model's sensitivity per input feature (0% = the feature is ignored):",
              f"{'Qubit: feature':<36}" + "".join(f"{n[:26]:>28}" for n in runs)]
    for i, fn in enumerate(D["sel_names"]):
        lines.append(f"{('q%d: %s' % (i, fn)):<36}" + "".join(f"{r['influence'][i]:>28.1f}" for r in runs.values()))
    report = "\n".join(lines)
    print("\n" + report)
    return runs, report
