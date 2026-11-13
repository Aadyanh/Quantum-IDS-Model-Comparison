import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOCAL_DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(RESULTS_DIR, exist_ok=True)

CFG = dict(
    DATA_DIR='/kaggle/input',        # searched recursively for .parquet / .csv files that have a Label column
    DATA_FILES=None,              # or a list of file paths to force specific files
    MAX_SAMPLES=None,             # None = ALL rows; or e.g. 500_000 for a quick test
    DROP_DUPLICATES=True,         # CICIDS2017 has many duplicate flows; removing them avoids train/test leakage
    N_QUBITS=6,                   # 2**6 = 64 amplitudes = 64 features
    N_LAYERS=4,
    EPOCHS=30,
    BATCH_SIZE=8192,
    LR=0.05,
    N_RESTARTS=3,                 # random re-initialisations; best one (on validation) is kept
    TIME_BUDGET_MIN=120,          # training stops gracefully after this many minutes
    SEED=42,
)
