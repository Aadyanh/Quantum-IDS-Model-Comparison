import time
import math
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve


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


def evaluate_all(fitted, D):
    """Threshold tuned on validation (max F1); metrics on validation and on the OFFICIAL test set. Returns (runs, report)."""
    # ---- evaluate: threshold tuned on validation; metrics on validation and on the OFFICIAL test set ----
    in_seen = np.array([lab in D["seen"] for lab in D["lab_te"]])
    fam_groups = ["Normal", "DoS", "Probe", "R2L", "U2R"] + (["Other"] if (D["fam_te"] == "Other").any() else [])
    runs = {}
    for n, (m, fit_t) in fitted.items():
        t0 = time.time()
        sv, st = m.decision_function(D["Xva"]), m.decision_function(D["Xte"])
        infer_t = time.time() - t0
        thr = best_f1_threshold(D["yva"], sv)
        r = dict(test_scores=st, thr=thr, val=metrics_at(D["yva"], sv, thr), test=metrics_at(D["yte"], st, thr),
                 test_default=metrics_at(D["yte"], st, 0.0), fit_time=fit_t, infer_time=infer_t,
                 n_sv=int(m.n_support_.sum()) if hasattr(m, "n_support_") else 0)
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
        out = [f"{'Model':<52}" + "".join(f"{c:>10}" for c in cols)]
        for n, r in runs.items():
            out.append(f"{n:<52}" + "".join(f"{r[key][c]:>10.4f}" for c in cols))
        return "\n".join(out)
    groups = fam_groups + ["Seen attacks", "Novel attacks"]
    n_te = {f: int((D["fam_te"] == f).sum()) for f in fam_groups}
    n_te["Seen attacks"] = int(((D["yte"] == 1) & in_seen).sum())
    n_te["Novel attacks"] = int(((D["yte"] == 1) & ~in_seen).sum())
    lines = ["=" * 110, "OFFICIAL TEST SET (KDDTest+) - threshold tuned on validation (max F1)", "=" * 110, table("test"), "",
             "OFFICIAL TEST SET - default threshold (decision function >= 0)", table("test_default"), "",
             "VALIDATION (held-out 15% of KDDTrain+, same distribution as training) - tuned threshold", table("val"), "",
             f"{'Cost':<52}{'fit (s)':>10}{'infer (s)':>11}{'support vec.':>14}"]
    for n, r in runs.items():
        lines.append(f"{n:<52}{r['fit_time']:>10.1f}{r['infer_time']:>11.2f}{r['n_sv']:>14,}")
    lines += ["", "Detection rate per class on the OFFICIAL TEST SET (Normal = correctly kept normal):",
              f"{'Class':<16}{'n':>8}" + "".join(f"{n[:30]:>32}" for n in runs)]
    fmt = lambda v: f"{v:>32.4f}" if v == v else f"{'n/a':>32}"
    for gname in groups:
        lines.append(f"{gname:<16}{n_te[gname]:>8,}" + "".join(fmt(r["det"][gname]) for r in runs.values()))
    report = "\n".join(lines)
    print("\n" + report)
    return runs, report
