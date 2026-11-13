import os
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from src.config import CFG
from src.preprocess import load_and_preprocess_ton_iot
from src.hybrid_model import HybridQNN, get_device, wrap_parallel, selftest
from src.train import train_hybrid_model
from src.evaluate import evaluate_model
from src.visualize_data import generate_visualizations
from src.cudaq_worker import verify_with_cudaq


def main():
    print("==========================================")
    print(" V4: Hybrid QNN on TON-IoT (PyTorch GPU)")
    print("==========================================")
    np.random.seed(CFG["SEED"])
    torch.manual_seed(CFG["SEED"])
    device = get_device()
    selftest()

    # 1. Load data (FULL dataset)
    Xtr, Xva, Xte, ytr, yva, yte, input_dim = load_and_preprocess_ton_iot(CFG)

    # 2. Initialize the Hybrid Model
    model = HybridQNN(input_dim, CFG["N_QUBITS"], CFG["N_LAYERS"], CFG["HIDDEN"]).to(device)
    net = wrap_parallel(model, CFG["USE_MULTI_GPU"])

    # 3. Train
    model, hist = train_hybrid_model(model, net, Xtr, ytr, Xva, yva, CFG, device)

    # 4. Evaluate & visualize (held-out test split)
    y_true, y_pred, y_prob, metrics = evaluate_model(net, Xva, yva, Xte, yte, CFG, device)
    generate_visualizations(y_true, y_pred, y_prob, metrics, hist)

    # 5. Optional: cross-check the trained circuit with real CUDA-Q (separate process)
    if CFG["CUDAQ_CHECK"]:
        verify_with_cudaq(model, Xte, device)

    print("\nPipeline Complete! Check the results/ directory.")


if __name__ == "__main__":
    main()
