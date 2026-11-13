import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from src.config import RESULTS_DIR, LOCAL_DATA_DIR


def _unique(cols):
    seen, out = {}, []
    for c in cols:
        if c in seen:
            seen[c] += 1
            out.append(f"{c}.{seen[c]}")
        else:
            seen[c] = 0
            out.append(c)
    return out


def _columns(f):
    if f.lower().endswith(".parquet"):
        import pyarrow.parquet as pq
        return [c.strip() for c in pq.read_schema(f).names]
    return [c.strip() for c in pd.read_csv(f, nrows=0).columns]


def _read_any(f):
    df = pd.read_parquet(f) if f.lower().endswith(".parquet") else pd.read_csv(f, low_memory=False)
    df.columns = _unique([str(c).strip() for c in df.columns])
    return df


def find_data_files(cfg):
    files, seen = [], set()
    for root in [LOCAL_DATA_DIR, cfg["DATA_DIR"]]:
        if not os.path.isdir(root):
            continue
        found = (sorted(glob.glob(os.path.join(root, "**", "*.parquet"), recursive=True)) +
                 sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True)))
        for f in found:
            key = (os.path.basename(f).lower(), os.path.getsize(f))   # same file in two places is used once
            if key not in seen:
                seen.add(key)
                files.append(f)
    ok = []
    for f in files:
        try:
            if "label" in [c.lower() for c in _columns(f)]:
                ok.append(f)
        except Exception:
            continue
    if not ok:
        raise FileNotFoundError("No parquet/CSV with a Label column found. Put the CICIDS2017 parquet/CSV files in data/ (see README for the download link).")
    return ok


def load_dataset(cfg):
    """Loads + cleans ALL files. Returns raw float32 features, binary labels, attack names, feature names."""
    files = cfg["DATA_FILES"] or find_data_files(cfg)
    print("Using files:")
    for f in files:
        print("  ", f)
    df = pd.concat([_read_any(f) for f in files], ignore_index=True)
    lab_col = [c for c in df.columns if c.lower() == "label"][0]
    df[lab_col] = df[lab_col].astype(str).str.strip().str.upper()
    print(f"Rows loaded: {len(df):,}")

    if cfg["DROP_DUPLICATES"]:
        n0 = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"Dropped {n0 - len(df):,} duplicate rows -> {len(df):,}")

    y = (df[lab_col] != "BENIGN").astype(int).values
    names_all = df[lab_col].to_numpy()

    if cfg["MAX_SAMPLES"] and len(df) > cfg["MAX_SAMPLES"]:
        keep, _ = train_test_split(np.arange(len(y)), train_size=cfg["MAX_SAMPLES"],
                                   stratify=y, random_state=cfg["SEED"])
        df, y, names_all = df.iloc[keep].reset_index(drop=True), y[keep], names_all[keep]
    print("Class counts:", dict(pd.Series(names_all).value_counts()))
    print(f"Rows used: {len(y):,} | attack share: {y.mean():.3f}")

    drop_l = {"label", "flow id", "source ip", "src ip", "source port", "src port", "destination ip",
              "dst ip", "destination port", "dst port", "timestamp", "ts", "type"}
    X = df.drop(columns=[c for c in df.columns if c.lower() in drop_l])
    del df
    for c in X.columns:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    feat_names = list(X.columns)
    arr = X.to_numpy(dtype=np.float32)
    del X
    arr[~np.isfinite(arr)] = 0.0
    print(f"Raw feature matrix: {arr.shape}")
    return arr, y, names_all, feat_names


def select_features(arr, y, tr, feat_names, dim, seed):
    """Top-`dim` features by Random Forest importance, using TRAIN rows only (no leakage)."""
    samp = np.random.RandomState(seed).choice(tr, size=min(100_000, len(tr)), replace=False)
    rf = RandomForestClassifier(n_estimators=50, min_samples_leaf=2, n_jobs=-1, random_state=seed)
    rf.fit(arr[samp], y[samp])
    sel = np.argsort(-rf.feature_importances_)[:dim]
    sel_names = [feat_names[i] for i in sel]
    print(f"Selected {len(sel)} features (top 5: {sel_names[:5]})")

    imp = pd.Series(rf.feature_importances_, index=feat_names).sort_values(ascending=False)
    plots = [
        ("all_feature_importance.png", imp, "All Feature Importances (Random Forest)", "steelblue"),
        ("top_64_features.png", imp.head(dim), f"Top {dim} Feature Importances (amplitude encoding)", "mediumseagreen"),
        ("feature_importance.png", imp.head(15), "Top 15 Feature Importances", "dodgerblue"),
    ]
    for fname, s, title, col in plots:
        plt.figure(figsize=(12 if len(s) > 15 else 8, 5))
        s.plot(kind="bar", color=col)
        plt.title(title)
        if len(s) > 15:
            plt.xticks([])
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, fname), dpi=200)
        plt.close()
    return sel, sel_names
