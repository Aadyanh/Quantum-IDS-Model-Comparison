import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOCAL_DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(RESULTS_DIR, exist_ok=True)

CFG = dict(
    DATA_DIR='/kaggle/input',   # folder searched (recursively) for UNSW-NB15_1..4.csv, besides ./data
    MAX_SAMPLES=None,        # None = ALL rows; or e.g. 300_000 for a quick test
    DROP_DUPLICATES=True,    # remove duplicate flows (avoids train/test leakage)
    N_QUBITS=6,              # = number of features (one feature per qubit)
    ENC_SCALES=[0.5, 1.0, 1.5],  # angle = pi * scale * x ; best scale chosen on validation
    EPOCHS=30,
    BATCH=8192,
    LR=0.05,
    L2=1e-6,                 # SVM regularisation (lambda/2 * ||W||^2)
    RUN_BASELINES=True,      # classical linear SVM + small MLP on the same features
    CUDAQ_CHECK=True,        # verify the kernel against CUDA-Q at the end
    SEED=42,
)
