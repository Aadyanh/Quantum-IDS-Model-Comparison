import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import numpy as np

from src.config import CFG, RESULTS_DIR
from src.preprocess import load_dataset
from src.split_data import get_split
from src.train_svm import train_models
from src.evaluate import evaluate_all
from src.visualize_data import generate_visualizations


def main():
    print("=" * 60)
    print("   V5: Classical RBF SVM Baseline (NSL-KDD)")
    print("=" * 60)
    np.random.seed(CFG["SEED"])

    print("\n[1/4] Loading KDDTrain and KDDTest...")
    tr, te = load_dataset()

    print("\n[2/4] Splitting, selecting features, scaling...")
    D = get_split(tr, te)

    print("\n[3/4] Training the SVMs on ALL training rows...")
    fitted, grid = train_models(D)

    print("\n[4/4] Evaluating (validation + official test set) and plotting...")
    runs, report = evaluate_all(fitted, D)
    generate_visualizations(runs, D, grid)
    with open(os.path.join(RESULTS_DIR, "final_results.txt"), "w") as f:
        f.write(report + "\n")
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump({n: {"test": {k: v for k, v in r["test"].items() if k != "pred"},
                       "val": {k: v for k, v in r["val"].items() if k != "pred"},
                       "detection": {k: float(v) for k, v in r["det"].items()},
                       "fit_time_s": r["fit_time"], "infer_time_s": r["infer_time"], "support_vectors": r["n_sv"]}
                  for n, r in runs.items()}, f, indent=2)

    print("\nPipeline complete. Check the results/ directory.")


if __name__ == "__main__":
    main()
