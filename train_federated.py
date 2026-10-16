import argparse
import os
import pickle
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

from aggregator import TrustAwareAggregator
from blockchain import TrustAwareBlockchain
from federated_node import FederatedNode
from partition import partition_dirichlet

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def eval_model(model, X, y, device):
    model.eval()
    xb = torch.tensor(X, dtype=torch.float32, device=device)
    logits = model(xb)
    probs = torch.sigmoid(logits).cpu().numpy().flatten()
    pred = (probs >= 0.5).astype(int)
    acc = accuracy_score(y, pred)
    try:
        auc = roc_auc_score(y, probs)
    except ValueError:
        auc = float("nan")
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits.cpu(), torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    ).item()
    return acc, auc, loss

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dirichlet-alpha", type=float, default=0.5)
    p.add_argument("--num-nodes", type=int, default=10)
    p.add_argument("--num-attackers", type=int, default=0)
    p.add_argument("--trust-aware", action="store_true")
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--local-epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="saved/run")
    p.add_argument("--data", default="data/processed_v2.pkl")
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    with open(args.data, "rb") as f:
        data = pickle.load(f)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    # --- بخش جدید: داده‌های پاک برای ممیزی توسط سرور ---
    # ۵۰۰ نمونه از داده‌های اعتبار سنجی صرفاً برای سرور جدا می‌شود
    X_server_clean = X_val[:500]
    y_server_clean = y_val[:500]
    # --------------------------------------------------

    node_idx = partition_dirichlet(y_train, args.num_nodes, args.dirichlet_alpha, args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    nodes = []
    for i, ix in enumerate(node_idx):
        y_node = y_train[ix].copy()
        if i < args.num_attackers:
            y_node = 1 - y_node  
            print(f"[*] Node {i:02d} is MALICIOUS (Label Flipped).")
        nodes.append(FederatedNode(X_train[ix], y_node, device))

    aggregator = TrustAwareAggregator(device, patience=10)
    blockchain = TrustAwareBlockchain(
        authorities=["aggregator"] + [f"node{i}" for i in range(len(nodes))], 
        num_nodes=len(nodes)
    )

    loss_hist, acc_hist = [], []

    for rnd in range(args.rounds):
        print(f"\n--- Round {rnd + 1}/{args.rounds} ---")
        full = aggregator.full_state()
        for node in nodes:
            node.load_full_state(full)

        local_weights = []
        server_audited_losses = []
        
        for i, node in enumerate(nodes):
            out = node.train_local(epochs=args.local_epochs, lr=args.lr)
            local_weights.append(out["weights"])
            
            # --- بخش جدید: ممیزی وزن‌ها توسط سرور ---
            # قرار دادن موقت وزن کلاینت روی مدل سرور
            aggregator.global_model.load_state_dict(out["weights"])
            # محاسبه خطا روی داده‌های پاک سرور
            _, _, audited_loss = eval_model(aggregator.global_model, X_server_clean, y_server_clean, device)
            server_audited_losses.append(audited_loss)
            # بازگرداندن مدل به حالت اولیه برای کلاینت بعدی
            aggregator.global_model.load_state_dict(full)
            # ---------------------------------------
            
            blockchain.add_block({"event": "local_train", "node": i, "audited_loss": audited_loss}, signer=f"node{i}")

        print(f"Server Audited Losses: {['{:.4f}'.format(l) for l in server_audited_losses]}")
        
        # ارسال خطاهای ممیزی‌شده به بلاک‌چین به جای خطاهای گزارش‌شده توسط کلاینت
        trust_scores = blockchain.evaluate_trust(server_audited_losses)
        print(f"Current Trust Scores : {['{:.2f}'.format(s) for s in trust_scores]}")

        agg_weights = trust_scores if args.trust_aware else np.ones(len(nodes))
        aggregator.aggregate(local_weights, agg_weights)

        val_acc, val_auc, val_loss = eval_model(aggregator.global_model, X_val, y_val, device)
        test_acc, test_auc, _ = eval_model(aggregator.global_model, X_test, y_test, device)
        loss_hist.append(val_loss)
        acc_hist.append(test_acc)
        
        print(f"Val Loss: {val_loss:.4f} | Test Acc: {test_acc:.4f} | Test AUC: {test_auc:.4f}")
        
        if aggregator.check_early_stopping(val_loss, save_dir=args.out):
            print("Early stopping triggered!")
            break

if __name__ == "__main__":
    main()