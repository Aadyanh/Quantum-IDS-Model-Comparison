import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import QuantileTransformer
from src.config import LOCAL_DATA_DIR


def find_data_files(cfg):
    """All CSVs from data/ and /kaggle/input (same file found twice is used once), grouped by columns."""
    seen, files = set(), []
    for root in [LOCAL_DATA_DIR, cfg["DATA_DIR"]]:
        if not os.path.isdir(root):
            continue
        for f in sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)):
            key = (os.path.basename(f).lower(), os.path.getsize(f))
            if key in seen:
                continue
            seen.add(key)
            files.append(f)
    groups = {}
    for f in files:
        try:
            cols = tuple(c.strip() for c in pd.read_csv(f, nrows=0).columns)
        except Exception:
            continue
        if "label" in cols:
            groups.setdefault(cols, []).append(f)
    if not groups:
        raise FileNotFoundError("No CSV with a 'label' column found. Put train_test_network.csv in data/ (see README for the download link).")
    pref = [g for g in groups.values() if any("train_test" in os.path.basename(f).lower() for f in g)]
    pool = pref or list(groups.values())
    return max(pool, key=lambda g: sum(os.path.getsize(f) for f in g))


def load_and_preprocess_ton_iot(cfg):
    """Loads the FULL TON-IoT data, splits 80/10/10 (stratified), encodes + scales (fit on train only)."""
    print("Loading TON-IoT dataset...")
    files = cfg["DATA_FILES"] or find_data_files(cfg)
    print("Using files:")
    for f in files:
        print("  ", f)
    df = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    df.columns = df.columns.str.strip()
    y_raw = df["label"]
    y = pd.to_numeric(y_raw, errors="coerce")
    if y.isna().any():
        y = (~y_raw.astype(str).str.lower().isin(["0", "normal", "benign"])).astype(int)
    y = y.astype(int).values

    if cfg["MAX_SAMPLES"] and len(df) > cfg["MAX_SAMPLES"]:
        keep, _ = train_test_split(np.arange(len(y)), train_size=cfg["MAX_SAMPLES"],
                                   stratify=y, random_state=cfg["SEED"])
        df, y = df.iloc[keep].reset_index(drop=True), y[keep]
    print(f"Rows: {len(y):,} | attack share: {y.mean():.3f}")

    # 'type' is the attack category -> it leaks the label, so it must go (plus identifiers)
    drop = {"label", "type", "ts", "src_ip", "dst_ip", "src_port", "dst_port"}
    X = df.drop(columns=[c for c in df.columns if c in drop])
    del df

    num_cols, cat_cols = [], []
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]):
            num_cols.append(c)
        else:
            num = pd.to_numeric(X[c], errors="coerce")          # numeric columns polluted with '-'
            if num.notna().mean() >= 0.95:
                X[c] = num.fillna(0.0)
                num_cols.append(c)
            else:
                cat_cols.append(c)
    Xnum = X[num_cols].to_numpy(dtype=np.float32)
    Xnum[~np.isfinite(Xnum)] = 0.0
    cat_vals = {c: X[c].astype(str).to_numpy() for c in cat_cols}
    del X

    # stratified 80 / 10 / 10 split. Everything below is FIT ON TRAIN ONLY.
    idx = np.arange(len(y))
    tr, tmp = train_test_split(idx, test_size=0.2, stratify=y, random_state=cfg["SEED"])
    va, te = train_test_split(tmp, test_size=0.5, stratify=y[tmp], random_state=cfg["SEED"])

    specs = []
    for c in cat_cols:
        vc = pd.Series(cat_vals[c][tr]).value_counts()
        if len(vc) <= 15:
            specs.append((c, "onehot", list(vc.index)))
        else:
            specs.append((c, "freq", (vc / vc.sum()).to_dict()))

    def build(ix):
        scal, onehot = [Xnum[ix]], []
        for c, kind, info in specs:
            v = cat_vals[c][ix]
            if kind == "freq":
                scal.append(pd.Series(v).map(info).fillna(0).to_numpy(np.float32)[:, None])
            else:
                codes = pd.Categorical(v, categories=info).codes
                onehot.append(np.eye(len(info) + 1, dtype=np.float32)[codes][:, :len(info)])
        return np.hstack(scal), (np.hstack(onehot) if onehot else np.zeros((len(ix), 0), np.float32))

    S_tr, O_tr = build(tr)
    S_va, O_va = build(va)
    S_te, O_te = build(te)
    qt = QuantileTransformer(n_quantiles=min(1000, len(tr)), output_distribution="uniform",
                             subsample=500_000, random_state=cfg["SEED"]).fit(S_tr)
    fin = lambda S, O: np.clip(np.hstack([qt.transform(S), O]), 0, 1).astype(np.float32)
    Xtr, Xva, Xte = fin(S_tr, O_tr), fin(S_va, O_va), fin(S_te, O_te)

    print("--- Data Preprocessed Successfully ---")
    print(f"Classical Input Features: {Xtr.shape[1]}")
    print(f"train {len(tr):,} | val {len(va):,} | test {len(te):,}")
    return Xtr, Xva, Xte, y[tr], y[va], y[te], Xtr.shape[1]
