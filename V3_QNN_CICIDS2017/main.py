import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import CFG
from src.preprocess import load_dataset
from src.split_data import split_dataset
from src.quantum_model import get_device, selftest
from src.train_qnn import train_qnn
from src.evaluate import evaluate_model
from src.visualize_data import generate_visualizations


def main():
    print("========================================")
    print(" V3: QNN on CICIDS2017 (Amplitude Enc.) ")
    print("========================================")
    device = get_device()
    selftest()

    print("\n[1/5] Loading and cleaning the FULL dataset...")
    arr, y, names, feat_names = load_dataset(CFG)

    print("\n[2/5] Splitting, selecting features, scaling, L2 normalisation...")
    Xtr, Xva, Xte, ytr, yva, yte, names_te = split_dataset(arr, y, names, feat_names, CFG)
    del arr

    print("\n[3/5] Training Quantum Neural Network (GPU)...")
    model, hist = train_qnn(Xtr, ytr, Xva, yva, CFG, device)

    print("\n[4/5] Evaluating on the held-out test split...")
    y_pred, y_prob, metrics = evaluate_model(model, Xva, yva, Xte, yte, names_te, CFG, device)

    print("\n[5/5] Visualizing results...")
    generate_visualizations(yte, y_pred, y_prob, metrics, hist, CFG)

    print("\nPipeline Complete! Check the results/ directory for your research outputs.")


if __name__ == "__main__":
    main()
