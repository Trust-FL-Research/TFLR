import argparse
import os
import pickle

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from core.model import CNN_LSTM_Model


def metrics(y_true, probs):
    pred = (probs >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "auc": roc_auc_score(y_true, probs),
        "report": classification_report(y_true, pred, zero_division=0),
    }


def eval_state(state_path, X, y, device):
    model = CNN_LSTM_Model().to(device)
    model.load_state_dict(torch.load(state_path, map_location=device))
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32, device=device))
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
    return metrics(y, probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="saved/run")
    parser.add_argument("--data", default="data/processed_v2.pkl")
    args = parser.parse_args()

    with open(args.data, "rb") as f:
        data = pickle.load(f)
    X_test, y_test = data["X_test"], data["y_test"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    global_path = os.path.join(args.run, "model_global.pth")
    best_path = os.path.join(args.run, "model_best.pth")
    target = best_path if os.path.exists(best_path) else global_path
    print("Evaluating", target)
    res = eval_state(target, X_test, y_test, device)
    print("\n===== Global / shared+avg-head =====")
    print(f"Accuracy: {res['accuracy']:.4f}")
    print(f"Precision: {res['precision']:.4f}")
    print(f"Recall: {res['recall']:.4f}")
    print(f"F1 Score: {res['f1']:.4f}")
    print(f"AUC: {res['auc']:.4f}")
    print(res["report"])

    client_files = sorted(
        f for f in os.listdir(args.run) if f.startswith("client_") and f.endswith(".pth")
    )
    if client_files:
        accs = []
        print("===== Personalized clients on global test =====")
        for fname in client_files:
            path = os.path.join(args.run, fname)
            cres = eval_state(path, X_test, y_test, device)
            accs.append(cres["accuracy"])
            print(f"{fname}: acc={cres['accuracy']:.4f} f1={cres['f1']:.4f} auc={cres['auc']:.4f}")
        print(f"client mean acc={np.mean(accs):.4f} std={np.std(accs):.4f}")


if __name__ == "__main__":
    main()
