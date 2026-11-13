import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import CFG, LOCAL_DATA_DIR

COLUMN_NAMES = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur",
    "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss", "service",
    "Sload", "Dload", "Spkts", "Dpkts", "swin", "dwin", "stcpb", "dtcpb",
    "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "Sjit", "Djit",
    "Stime", "Ltime", "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat",
    "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login",
    "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "attack_cat", "label"
]
CAT_COLS = ["proto", "service", "state"]
# identifiers + timestamps (timestamps leak the label in a random split)
DROP_COLS = ["srcip", "dstip", "attack_cat", "label", "Stime", "Ltime"]


def find_files():
    files, seen = [], set()
    for root in [LOCAL_DATA_DIR, CFG["DATA_DIR"]]:
        if not root or not os.path.isdir(root):
            continue
        for f in sorted(glob.glob(os.path.join(root, "**", "UNSW-NB15_[0-9]*.csv"), recursive=True)):
            key = (os.path.basename(f), os.path.getsize(f))      # same file in two places is used once
            if key not in seen:
                seen.add(key)
                files.append(f)
    if not files:
        raise FileNotFoundError("UNSW-NB15_1..4.csv not found. Put them in data/ (see README for the download link).")
    return sorted(files, key=os.path.basename)


def load_dataset():
    files = find_files()
    print("Using files:")
    for f in files:
        print("  ", f)
    dfs = []
    for f in files:
        df = pd.read_csv(f, header=None, names=COLUMN_NAMES, low_memory=False)
        dfs.append(df[df["srcip"].astype(str).str.strip().str.lower() != "srcip"])
    data = pd.concat(dfs, ignore_index=True)
    del dfs
    print(f"Rows loaded: {len(data):,}")

    data["label"] = pd.to_numeric(data["label"], errors="coerce").fillna(0).astype(int)
    cat = data["attack_cat"].astype(str).str.strip().replace({"nan": "Unspecified", "": "Unspecified",
                                                               "Backdoors": "Backdoor"})
    data["attack_cat"] = cat.where(data["label"] == 1, "Normal")

    if CFG["DROP_DUPLICATES"]:
        n0 = len(data)
        subset = [c for c in COLUMN_NAMES if c not in ("srcip", "dstip", "Stime", "Ltime", "attack_cat")]
        data = data.drop_duplicates(subset=subset).reset_index(drop=True)
        print(f"Dropped {n0 - len(data):,} duplicate rows -> {len(data):,}")

    if CFG["MAX_SAMPLES"] and len(data) > CFG["MAX_SAMPLES"]:
        keep, _ = train_test_split(np.arange(len(data)), train_size=CFG["MAX_SAMPLES"],
                                   stratify=data["label"].values, random_state=CFG["SEED"])
        data = data.iloc[np.sort(keep)].reset_index(drop=True)

    y = data["label"].to_numpy(np.int64)
    names = data["attack_cat"].to_numpy()
    print(f"Rows used: {len(y):,} | attack share: {y.mean():.3f}")
    print("Classes:", dict(pd.Series(names).value_counts()))

    X = data.drop(columns=DROP_COLS)
    del data
    num_cols = [c for c in X.columns if c not in CAT_COLS]
    for c in num_cols:                                    # ports / blanks / '-' -> NaN -> 0
        X[c] = pd.to_numeric(X[c], errors="coerce")
    num = X[num_cols].fillna(0.0).to_numpy(np.float32)
    num[~np.isfinite(num)] = 0.0
    cats = {c: X[c].astype(str).str.strip().to_numpy() for c in CAT_COLS}
    del X
    return num, num_cols, cats, y, names
