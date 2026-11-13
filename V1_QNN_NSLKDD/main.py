import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np
import torch

from src.config import CFG, RESULTS_DIR
from src.preprocess import load_dataset
from src.split_data import get_split
from src.train_binary import train_all
from src.evaluate import evaluate_all
from src.quantum_classifier import cudaq_check
from src.visualize_data import generate_visualizations


def main():
    print("=" * 60)
    print("        V1: Quantum Intrusion Classifier (NSL-KDD)")
    print("=" * 60)
    torch.manual_seed(CFG["SEED"])
    np.random.seed(CFG["SEED"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, "|", ", ".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())))
    if device.type == "cpu":
        print("WARNING: no GPU found - turn on a GPU accelerator in the notebook settings.")

    print("\n[1/5] Loading KDDTrain and KDDTest...")
    tr, te = load_dataset()

    print("\n[2/5] Splitting, selecting features, scaling...")
    D = get_split(tr, te)

    print("\n[3/5] Training the QNNs (GPU) and the classical baselines...")
    models = train_all(D, device)

    print("\n[4/5] Evaluating (validation + official test set)...")
    runs, report = evaluate_all(models, D, device)

    print("\n[5/5] Visualizing results + CUDA-Q circuit check...")
    generate_visualizations(runs, D)
    best_qnn = next(iter(models.values()))[0]
    torch.save(best_qnn.state_dict(), os.path.join(RESULTS_DIR, "qnn_improved.pt"))
    np.save(os.path.join(RESULTS_DIR, "optimized_theta.npy"), best_qnn.theta.detach().cpu().numpy())
    check = cudaq_check(best_qnn, D["Xte"]) if CFG["CUDAQ_CHECK"] else "disabled"
    with open(os.path.join(RESULTS_DIR, "final_results.txt"), "w") as f:
        f.write(report + "\n\nCUDA-Q circuit check: " + check + "\n")
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump({n: {"test": {k: v for k, v in r["test"].items() if k != "pred"},
                       "val": {k: v for k, v in r["val"].items() if k != "pred"},
                       "detection": {k: float(v) for k, v in r["det"].items()},
                       "influence_pct": [float(v) for v in r["influence"]],
                       "history": r["hist"]} for n, r in runs.items()}, f, indent=2)

    print("\nPipeline Complete! Check the results/ directory for your research outputs.")


if __name__ == "__main__":
    main()
