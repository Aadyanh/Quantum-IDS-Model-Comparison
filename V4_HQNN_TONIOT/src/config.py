import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOCAL_DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(RESULTS_DIR, exist_ok=True)

CFG = dict(
    DATA_DIR="/kaggle/input",     # also searched (recursively) for CSVs, after the project's own data/ folder
    DATA_FILES=None,              # or a list of CSV paths to force specific files
    MAX_SAMPLES=None,             # None = use ALL rows; or e.g. 1_000_000 for a stratified subset
    N_QUBITS=16,
    N_LAYERS=2,                   # variational layers (RX,RZ per qubit + CNOT chain)
    HIDDEN=64,                    # classical hidden layer before the quantum layer; 0 = single Linear (original)
    EPOCHS=8,
    BATCH_PER_GPU=256,
    LR=5e-3,
    USE_MULTI_GPU=True,           # DataParallel if more than one GPU (e.g. a Kaggle "T4 x2" machine)
    TIME_BUDGET_MIN=480,          # training stops gracefully after this many minutes
    SEED=42,
    CUDAQ_CHECK=True,             # cross-check the circuit with CUDA-Q at the end (skipped if not installed)
)
