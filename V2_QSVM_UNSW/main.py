import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

from src.config import CFG, RESULTS_DIR
from src.preprocess import load_dataset
from src.split_data import split_dataset
from src.train_qsvm import train_all
from src.evaluate import evaluate_all
from src.quantum_kernel import cudaq_check
from src.visualize_data import generate_visualizations


def main():
    print("========================================")
    print(" V2: QSVM on UNSW-NB15 (Quantum Kernel) ")
    print("========================================")
    torch.manual_seed(CFG["SEED"])
    np.random.seed(CFG["SEED"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, "|", ", ".join(torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())))
    if device.type == "cpu":
        print("WARNING: no GPU found - turn on a GPU accelerator in the notebook settings.")

    print("\n[1/5] Loading and cleaning the FULL dataset...")
    num, num_cols, cats, y, names = load_dataset()

    print("\n[2/5] Splitting, selecting features, scaling...")
    D = split_dataset(num, num_cols, cats, y, names)
    del num, cats

    print("\n[3/5] Training the QSVM (GPU) and the classical baselines...")
    models, search = train_all(D, device)

    print("\n[4/5] Evaluating on the held-out test split...")
    runs, report = evaluate_all(models, D, device)

    print("\n[5/5] Visualizing results + CUDA-Q kernel check...")
    generate_visualizations(runs, D, search)
    qsvm = next(iter(models.values()))[0]
    torch.save(qsvm.state_dict(), os.path.join(RESULTS_DIR, "qsvm.pt"))
    check = cudaq_check(qsvm, D["Xte"]) if CFG["CUDAQ_CHECK"] else "disabled"
    with open(os.path.join(RESULTS_DIR, "final_results.txt"), "a") as f:
        f.write("\nCUDA-Q kernel check: " + check + "\n")

    print("\nPipeline Complete! Check the results/ directory for your research outputs.")


if __name__ == "__main__":
    main()
