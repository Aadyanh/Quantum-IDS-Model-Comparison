import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
LOCAL_DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(RESULTS_DIR, exist_ok=True)

CFG = dict(
    DATA_DIR='/kaggle/input',   # folder searched (recursively) for KDDTrain / KDDTest (.parquet, .txt or .csv), besides ./data
    N_QUBITS=8,              # = number of features (one feature per qubit)
    N_LAYERS=4,              # [CNOT chain + RY layer] blocks of the improved QNN (original had 2)
    EPOCHS=60,
    BATCH=2048,
    LR=0.02,
    N_RESTARTS=3,            # random re-initialisations of the QNNs; best on validation is kept
    VAL_FRACTION=0.15,       # share of KDDTrain kept for validation / threshold tuning
    RUN_ORIGINAL_STYLE=True, # also train the original design (2 layers, mean of Z0..Z3 readout)
    RUN_BASELINES=True,      # classical logistic regression + small MLP on the same features
    CUDAQ_CHECK=True,        # verify the simulator against CUDA-Q at the end
    SEED=42,
)
