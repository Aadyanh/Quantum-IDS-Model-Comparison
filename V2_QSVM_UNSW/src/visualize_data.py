import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve
from src.config import RESULTS_DIR as OUT


def save_fig(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, name), dpi=200, bbox_inches="tight")
    plt.close()


def generate_visualizations(runs, D, search):
    pal = plt.cm.tab10.colors
    col = {n: pal[i] for i, n in enumerate(runs)}
    yte = D["yte"]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for n, r in runs.items():
        ax[0].plot(r["hist"]["train_loss"], color=col[n], label=n)
        ax[1].plot(r["hist"]["val_ap"], color=col[n], label=n)
    ax[0].set_title("Training loss"); ax[0].set_xlabel("Epoch"); ax[0].legend()
    ax[1].set_title("Validation PR-AUC"); ax[1].set_xlabel("Epoch"); ax[1].legend()
    save_fig("training_curves.png")

    plt.figure(figsize=(6, 4))
    plt.bar([str(s) for s, _ in search], [a for _, a in search], color="tab:blue")
    for i, (_, a) in enumerate(search):
        plt.text(i, a + 0.002, f"{a:.4f}", ha="center")
    plt.xlabel("Encoding scale"); plt.ylabel("Best validation PR-AUC"); plt.title("QSVM encoding-scale search")
    save_fig("scale_search.png")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for n, r in runs.items():
        fpr, tpr, _ = roc_curve(yte, r["test_scores"])
        p, rc, _ = precision_recall_curve(yte, r["test_scores"])
        ax[0].plot(fpr, tpr, color=col[n], label=f"{n} (AUC {r['tuned']['roc_auc']:.4f})")
        ax[1].plot(rc, p, color=col[n], label=f"{n} (AP {r['tuned']['pr_auc']:.4f})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=1)
    ax[0].set_title("ROC (test)"); ax[0].set_xlabel("FPR"); ax[0].set_ylabel("TPR"); ax[0].legend(loc="lower right")
    ax[1].set_title("Precision-Recall (test)"); ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].legend(loc="lower left")
    save_fig("roc_pr.png")

    fig, ax = plt.subplots(1, len(runs), figsize=(4.8 * len(runs), 4.4), squeeze=False)
    for a, (n, r) in zip(ax[0], runs.items()):
        m = r["tuned"]
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        a.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            a.text(j, i, f"{v:,}", ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
        a.set_xticks([0, 1]); a.set_xticklabels(["Normal", "Attack"])
        a.set_yticks([0, 1]); a.set_yticklabels(["Normal", "Attack"])
        a.set_xlabel("Predicted"); a.set_ylabel("Actual"); a.set_title(f"{n}\n(test, tuned threshold)", fontsize=10)
    save_fig("confusion_matrix.png")

    keys = ["accuracy", "precision", "recall", "f1", "mcc", "roc_auc", "pr_auc"]
    plt.figure(figsize=(11, 4.5))
    w = 0.8 / len(runs)
    for k, (n, r) in enumerate(runs.items()):
        vals = [r["tuned"][m] for m in keys]
        plt.bar(np.arange(len(keys)) + k * w, vals, w, label=n, color=col[n])
        for i, v in enumerate(vals):
            plt.text(i + k * w, v + 0.01, f"{v:.3f}", ha="center", fontsize=7, rotation=90)
    plt.xticks(np.arange(len(keys)) + w * (len(runs) - 1) / 2, keys)
    plt.ylim(0, 1.2); plt.title("Test metrics (threshold tuned on validation)"); plt.legend(loc="lower right")
    save_fig("evaluation_metrics.png")

    first = next(iter(runs.values()))
    pred = first["tuned"]["pred"]
    plt.figure(figsize=(6, 4.5))
    plt.bar(np.arange(2) - 0.2, [(yte == 0).sum(), (yte == 1).sum()], 0.4, label="Actual")
    plt.bar(np.arange(2) + 0.2, [(~pred).sum(), pred.sum()], 0.4, label="Predicted (QSVM)")
    plt.xticks([0, 1], ["Normal (0)", "Attack (1)"]); plt.title("Actual vs predicted (test)"); plt.legend()
    save_fig("actual_vs_predicted.png")

    sub = np.random.RandomState(0).choice(len(D["Xte"]), size=min(20000, len(D["Xte"])), replace=False)
    plt.figure(figsize=(9, 4.5))
    plt.boxplot(D["Xte"][sub])
    plt.xticks(range(1, len(D["sel_names"]) + 1), D["sel_names"], rotation=30); plt.ylabel("Scaled value [0, 1]"); plt.title("Normalised features fed to the qubits (test sample)")
    save_fig("normalized_features.png")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    D["imp"].head(15)[::-1].plot(kind="barh", ax=ax[0], color="dodgerblue"); ax[0].set_title("Top 15 feature importances")
    D["imp"].plot(kind="bar", ax=ax[1], color="steelblue"); ax[1].set_xticks([]); ax[1].set_title("All feature importances")
    save_fig("feature_importance.png")
