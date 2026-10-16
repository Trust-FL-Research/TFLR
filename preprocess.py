"""Fit imputers/encoders/scalers on TRAIN only. Drop timestamp to reduce time leakage."""
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

CSV_CANDIDATES = [
    "data/ton-iot.csv",
    "data/ToN-IoT.csv",
    "data/Train_Test_Network.csv",
]

DROP_COLUMNS = [
    "ts",
    "src_ip",
    "dst_ip",
    "dns_query",
    "ssl_subject",
    "ssl_issuer",
    "http_uri",
    "http_user_agent",
    "http_orig_mime_types",
    "http_resp_mime_types",
    "http_referrer",
    "weird_addl",
    "weird_notice",
]

CATEGORICAL_COLS = [
    "proto",
    "service",
    "conn_state",
    "http_method",
    "http_version",
    "ssl_version",
    "ssl_cipher",
    "weird_name",
    "dns_AA",
    "dns_RD",
    "dns_RA",
    "dns_rejected",
    "ssl_resumed",
    "ssl_established",
]


def find_csv():
    for path in CSV_CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("CSV not found. Put the ToN-IoT network file in data/ton-iot.csv")


def present(cols, frame):
    return [c for c in cols if c in frame.columns]


def safe_label_transform(series, encoder):
    classes = set(encoder.classes_)
    fallback = "unknown" if "unknown" in classes else encoder.classes_[0]
    mapped = series.astype(str).where(series.astype(str).isin(classes), fallback)
    return encoder.transform(mapped)


def main():
    csv_path = find_csv()
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"Original shape: {df.shape}")
    df = df.replace("-", np.nan)
    df = df.drop(columns=present(DROP_COLUMNS, df), errors="ignore")

    if "label" not in df.columns:
        raise ValueError("Column 'label' is required")

    y = df["label"].astype(int)
    extra_drop = [c for c in ["label", "type"] if c in df.columns]
    X_df = df.drop(columns=extra_drop)

    cat_cols = present(CATEGORICAL_COLS, X_df)
    num_cols = [c for c in X_df.columns if c not in cat_cols]
    print(f"Numeric cols ({len(num_cols)}): {num_cols}")
    print(f"Categorical cols ({len(cat_cols)}): {cat_cols}")

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_df, y, test_size=0.20, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.10, random_state=42, stratify=y_trainval
    )

    X_train = X_train.copy()
    X_val = X_val.copy()
    X_test = X_test.copy()

    for col in cat_cols:
        X_train[col] = X_train[col].fillna("unknown").astype(str)
        X_val[col] = X_val[col].fillna("unknown").astype(str)
        X_test[col] = X_test[col].fillna("unknown").astype(str)

    num_imputer = None
    scaler = None
    if num_cols:
        num_imputer = SimpleImputer(strategy="mean")
        X_train[num_cols] = num_imputer.fit_transform(X_train[num_cols])
        X_val[num_cols] = num_imputer.transform(X_val[num_cols])
        X_test[num_cols] = num_imputer.transform(X_test[num_cols])
        scaler = MinMaxScaler()
        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_val[num_cols] = scaler.transform(X_val[num_cols])
        X_test[num_cols] = scaler.transform(X_test[num_cols])

    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        X_val[col] = safe_label_transform(X_val[col], le)
        X_test[col] = safe_label_transform(X_test[col], le)
        encoders[col] = le

    def pack(frame, labels):
        arr = frame.to_numpy(dtype=np.float32).reshape(len(frame), frame.shape[1], 1)
        return arr, labels.to_numpy(dtype=np.int64)

    Xtr, ytr = pack(X_train, y_train)
    Xva, yva = pack(X_val, y_val)
    Xte, yte = pack(X_test, y_test)

    payload = {
        "X_train": Xtr,
        "y_train": ytr,
        "X_val": Xva,
        "y_val": yva,
        "X_test": Xte,
        "y_test": yte,
        "feature_names": list(X_train.columns),
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "dropped": present(DROP_COLUMNS, df) + extra_drop,
    }
    os.makedirs("data", exist_ok=True)
    out_path = "data/processed_v2.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"Features: {Xtr.shape[1]}")
    print(f"Train {Xtr.shape} attack_ratio={ytr.mean():.3f}")
    print(f"Val   {Xva.shape} attack_ratio={yva.mean():.3f}")
    print(f"Test  {Xte.shape} attack_ratio={yte.mean():.3f}")
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
