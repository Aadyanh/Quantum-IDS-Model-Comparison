import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
from src.config import RESULTS_DIR as OUT


def save_fig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, name), dpi=200, bbox_inches="tight")
    plt.close()


def generate_visualizations(runs, D):
    pal = plt.cm.tab10.colors
    col = {n: pal[i] for i, n in enumerate(runs)}
    yte = D["yte"]
    short = lambda n: n if len(n) < 34 else n[:31] + "..."

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for n, r in runs.items():
        ax[0].plot(r["hist"]["train_loss"], color=col[n], label=short(n))
        ax[1].plot(r["hist"]["val_ap"], color=col[n], label=short(n))
    ax[0].set_title("Training loss"); ax[0].set_xlabel("Epoch"); ax[0].legend(fontsize=8)
    ax[1].set_title("Validation PR-AUC"); ax[1].set_xlabel("Epoch"); ax[1].legend(fontsize=8)
    save_fig("training_curves.png")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for n, r in runs.items():
        fpr, tpr, _ = roc_curve(yte, r["test_scores"])
        p, rc, _ = precision_recall_curve(yte, r["test_scores"])
        ax[0].plot(fpr, tpr, color=col[n], label=f"{short(n)} (AUC {r['test']['roc_auc']:.3f})")
        ax[1].plot(rc, p, color=col[n], label=f"{short(n)} (AP {r['test']['pr_auc']:.3f})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=1)
    ax[0].set_title("ROC (official test set)"); ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR"); ax[0].legend(fontsize=8, loc="lower right")
    ax[1].set_title("Precision-Recall (official test set)"); ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].legend(fontsize=8, loc="lower left")
    save_fig("roc_pr.png")

    fig, ax = plt.subplots(1, len(runs), figsize=(4.6 * len(runs), 4.4), squeeze=False)
    for a, (n, r) in zip(ax[0], runs.items()):
        m = r["test"]
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        a.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            a.text(j, i, f"{v:,}", ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
        a.set_xticks([0, 1]); a.set_xticklabels(["Normal", "Attack"])
        a.set_yticks([0, 1]); a.set_yticklabels(["Normal", "Attack"])
        a.set_xlabel("Predicted"); a.set_ylabel("Actual"); a.set_title(short(n) + "\n(test, tuned threshold)", fontsize=9)
    save_fig("confusion_matrix.png")

    keys = ["accuracy", "precision", "recall", "f1", "mcc", "roc_auc", "pr_auc"]
    plt.figure(figsize=(12, 4.5))
    w = 0.8 / len(runs)
    for k, (n, r) in enumerate(runs.items()):
        vals = [r["test"][m] for m in keys]
        plt.bar(np.arange(len(keys)) + k * w, vals, w, label=short(n), color=col[n])
        for i, v in enumerate(vals):
            plt.text(i + k * w, v + 0.01, f"{v:.2f}", ha="center", fontsize=7, rotation=90)
    plt.xticks(np.arange(len(keys)) + w * (len(runs) - 1) / 2, keys)
    plt.ylim(0, 1.2); plt.title("Official test set metrics (threshold tuned on validation)"); plt.legend(fontsize=8, loc="lower right")
    save_fig("evaluation_metrics.png")

    groups = list(next(iter(runs.values()))["det"].keys())
    plt.figure(figsize=(12, 4.5))
    for k, (n, r) in enumerate(runs.items()):
        plt.bar(np.arange(len(groups)) + k * w, [r["det"][g] for g in groups], w, label=short(n), color=col[n])
    plt.xticks(np.arange(len(groups)) + w * (len(runs) - 1) / 2, groups)
    plt.ylim(0, 1.1); plt.ylabel("Detection rate (Normal: kept normal)"); plt.title("Per-class results on the official test set"); plt.legend(fontsize=8)
    save_fig("per_class_detection.png")

    qn = [n for n in runs if n.startswith("QNN")]
    plt.figure(figsize=(11, 4.5))
    w2 = 0.8 / max(len(qn), 1)
    for k, n in enumerate(qn):
        plt.bar(np.arange(len(D["sel_names"])) + k * w2, runs[n]["influence"], w2, label=short(n), color=col[n])
    plt.xticks(np.arange(len(D["sel_names"])) + w2 * (len(qn) - 1) / 2,
               [f"q{i}: {s}" for i, s in enumerate(D["sel_names"])], rotation=30, ha="right")
    plt.ylabel("Share of model sensitivity (%)"); plt.title("How much each qubit/feature influences the output"); plt.legend(fontsize=8)
    save_fig("feature_influence.png")

    first = next(iter(runs.values()))["test"]["pred"]
    plt.figure(figsize=(6, 4.5))
    plt.bar(np.arange(2) - 0.2, [(yte == 0).sum(), (yte == 1).sum()], 0.4, label="Actual")
    plt.bar(np.arange(2) + 0.2, [(~first).sum(), first.sum()], 0.4, label="Predicted (best QNN)")
    plt.xticks([0, 1], ["Normal", "Attack"]); plt.title("Actual vs predicted (test)"); plt.legend()
    save_fig("actual_vs_predicted.png")

    ytr_all, yte_all = D["counts"]["train"], D["counts"]["test"]
    plt.figure(figsize=(6, 4.5))
    plt.bar(np.arange(2) - 0.2, [(ytr_all == 0).sum(), (ytr_all == 1).sum()], 0.4, label="KDDTrain+")
    plt.bar(np.arange(2) + 0.2, [(yte_all == 0).sum(), (yte_all == 1).sum()], 0.4, label="KDDTest+")
    plt.xticks([0, 1], ["Normal", "Attack"]); plt.title("NSL-KDD class distribution"); plt.legend()
    save_fig("class_distribution.png")

    plt.figure(figsize=(9, 4.5))
    plt.boxplot(D["Xte"])
    plt.xticks(range(1, len(D["sel_names"]) + 1), D["sel_names"], rotation=30)
    plt.ylabel("Scaled value [0, 1]"); plt.title("Normalised features fed to the qubits (test set)")
    save_fig("normalized_features.png")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    D["imp"].head(15)[::-1].plot(kind="barh", ax=ax[0], color="dodgerblue"); ax[0].set_title("Top 15 feature importances")
    D["imp"].plot(kind="bar", ax=ax[1], color="steelblue"); ax[1].set_xticks([]); ax[1].set_title("All feature importances")
    save_fig("feature_importance.png")
