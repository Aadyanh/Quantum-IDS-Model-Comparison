import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOCAL_DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(RESULTS_DIR, exist_ok=True)

CFG = dict(
    DATA_DIR='/kaggle/input',   # folder searched (recursively) for KDDTrain / KDDTest (.parquet, .txt or .csv), besides ./data
    N_FEATURES=8,            # top features by Random Forest importance
    VAL_FRACTION=0.15,       # share of KDDTrain kept for validation / threshold tuning
    SEARCH=True,             # tune C and gamma on validation (grid below, on a subsample)
    SEARCH_ROWS=30_000,
    C_GRID=[1.0, 10.0, 100.0],
    GAMMA_MULTS=[0.5, 1.0, 2.0, 4.0],      # multiples of sklearn's gamma='scale'
    RUN_LINEAR=True,         # also a linear SVM as a reference
    SEED=42,
)
