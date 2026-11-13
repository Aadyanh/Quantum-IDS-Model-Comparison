import os
import re
import glob
import numpy as np
import pandas as pd
from src.config import CFG, LOCAL_DATA_DIR

COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty"
]
CAT_COLS = ["protocol_type", "service", "flag"]
ATTACK_FAMILY = {}
for _fam, _names in {
    "DoS": "back land neptune pod smurf teardrop apache2 mailbomb processtable udpstorm worm",
    "Probe": "ipsweep nmap portsweep satan mscan saint",
    "R2L": "ftp_write guess_passwd imap multihop phf spy warezclient warezmaster sendmail named snmpgetattack "
           "snmpguess xlock xsnoop httptunnel",
    "U2R": "buffer_overflow loadmodule perl rootkit ps sqlattack xterm",
}.items():
    for _n in _names.split():
        ATTACK_FAMILY[_n] = _fam


def find_data():
    roots = [r for r in [LOCAL_DATA_DIR, CFG["DATA_DIR"], "/kaggle/input"] if r and os.path.isdir(r)]
    tr = te = None
    seen_files = []
    for root in roots:
        for ext in ("parquet", "txt", "csv"):
            for f in sorted(glob.glob(os.path.join(root, "**", f"*.{ext}"), recursive=True)):
                seen_files.append(f)
                b = os.path.basename(f).lower()
                if tr is None and re.fullmatch(r"kddtrain[+_]?\.(parquet|txt|csv)", b):
                    tr = f
                if te is None and re.fullmatch(r"kddtest[+_]?\.(parquet|txt|csv)", b):
                    te = f
    if tr is None or te is None:
        raise FileNotFoundError("Could not find KDDTrain / KDDTest files (.parquet, .txt or .csv). Files seen: "
                                + (", ".join(seen_files[:20]) or "none - put KDDTrain/KDDTest in data/ (see README for the download link)"))
    return tr, te


LABEL_ALIASES = ["label", "labels", "attack", "attack_type", "attack_cat", "class", "outcome", "target"]
DROP_EXTRA = ["difficulty", "difficulty_level", "score"]


def read_nsl(f):
    """Reads parquet / csv / txt, normalises column names, builds df['label'] (names) and df['_y'] (0 normal / 1 attack)."""
    if f.lower().endswith(".parquet"):
        df = pd.read_parquet(f)
    else:
        with open(f, errors="ignore") as fh:
            first = fh.readline().split(",")[0].strip()
        try:
            float(first)
            header = None
        except ValueError:
            header = 0
        df = pd.read_csv(f, header=header)
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    if "src_bytes" not in df.columns and df.shape[1] in (42, 43):      # no header names -> use the standard ones
        df.columns = COLUMN_NAMES[:df.shape[1]]
    cands = [c for c in LABEL_ALIASES if c in df.columns]
    if not cands:
        raise ValueError(f"No label column found in {os.path.basename(f)}. Columns: {list(df.columns)}")
    lab_col = max(cands, key=lambda c: df[c].nunique())              # the most detailed label column
    sr = df[lab_col]
    if pd.api.types.is_numeric_dtype(sr):
        y = (sr.to_numpy() != 0).astype(np.int64)
        lab = np.where(y == 0, "normal", "attack_" + sr.astype(str).to_numpy())
        print(f"  [{os.path.basename(f)}] numeric label column '{lab_col}': assuming 0 = normal")
    else:
        lab = sr.astype(str).str.strip().str.lower().str.rstrip(".").to_numpy()
        y = (lab != "normal").astype(np.int64)
    print(f"  [{os.path.basename(f)}] {len(df):,} rows x {df.shape[1]} cols | label column '{lab_col}' "
          f"({len(set(lab))} distinct values)")
    df = df.drop(columns=[c for c in cands + DROP_EXTRA if c in df.columns])
    df["label"], df["_y"] = lab, y
    return df


def load_dataset():
    """Reads KDDTrain and KDDTest; returns two DataFrames with the same columns plus 'label' (name) and '_y' (0/1)."""
    f_tr, f_te = find_data()
    print("Train file:", f_tr, "\nTest file: ", f_te)
    tr, te = read_nsl(f_tr), read_nsl(f_te)
    return tr, te.reindex(columns=tr.columns)               # same columns / order as the training file
