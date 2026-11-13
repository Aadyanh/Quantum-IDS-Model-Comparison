import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from src.config import RESULTS_DIR


def _save_show(name):
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, name), dpi=200)


def generate_visualizations(y_true, y_pred, y_prob, metrics, hist):
    print("\nGenerating and saving graphs to results/ folder...")

    plt.figure(figsize=(7, 4.5))
    ea = np.arange(1, len(hist["train"]) + 1)
    plt.plot(ea, hist["train"], "o-", color="purple", label="train")
    plt.plot(ea, hist["val"], "s-", color="teal", label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Weighted BCE loss")
    plt.title("HQNN Training / Validation Loss")
    plt.legend()
    plt.grid(alpha=.3)
    _save_show("training_loss.png")

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.imshow(cm, cmap="Purples")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=12)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal (0)", "Attack (1)"])
    ax.set_yticklabels(["Normal (0)", "Attack (1)"])
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("Actual Class")
    ax.set_title("HQNN Confusion Matrix (test)")
    _save_show("confusion_matrix.png")

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure(figsize=(5.5, 4.8))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {metrics['roc_auc']:.4f})")
    plt.plot([0, 1], [0, 1], "--", color="navy")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")
    plt.grid(alpha=.3)
    _save_show("roc_curve.png")

    pp, rr, _ = precision_recall_curve(y_true, y_prob)
    plt.figure(figsize=(5.5, 4.8))
    plt.plot(rr, pp, color="green", lw=2, label=f"PR curve (AP = {metrics['pr_auc']:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(alpha=.3)
    _save_show("precision_recall_curve.png")

    names = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "PR-AUC"]
    vals = [metrics[k] for k in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]]
    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(names, vals, color=plt.cm.magma(np.linspace(.2, .8, 6)))
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v + .01, f"{v:.4f}", ha="center", fontweight="bold")
    plt.ylim(0, 1.1)
    plt.title("HQNN Evaluation Metrics (test)")
    _save_show("evaluation_metrics.png")

    print("All visualizations saved successfully!")
