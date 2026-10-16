import os
import torch
import numpy as np
from core.model import CNN_LSTM_Attention_Model

class TrustAwareAggregator:
    def __init__(self, device, patience=10):
        self.device = device
        self.global_model = CNN_LSTM_Attention_Model().to(device)
        self.best_loss = float("inf")
        self.no_improve = 0
        self.patience = patience

    def aggregate(self, client_weights, trust_scores):
        """Q1 Novelty: Trust-Weighted Aggregation"""
        new_state = {}
        all_keys = client_weights[0].keys()
        
        total_trust = np.sum(trust_scores)
        if total_trust == 0:
            weights = np.ones(len(client_weights)) / len(client_weights)
        else:
            weights = trust_scores / total_trust
            
        for key in all_keys:
            weighted_sum = None
            for i, client_w in enumerate(client_weights):
                w_tensor = client_w[key].float() * float(weights[i])
                if weighted_sum is None:
                    weighted_sum = w_tensor
                else:
                    weighted_sum += w_tensor
            new_state[key] = weighted_sum

        self.global_model.load_state_dict(new_state)

    def full_state(self):
        return {k: v.detach().cpu().clone() for k, v in self.global_model.state_dict().items()}

    def check_early_stopping(self, val_loss, save_dir="saved"):
        os.makedirs(save_dir, exist_ok=True)
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.no_improve = 0
            torch.save(self.global_model.state_dict(), os.path.join(save_dir, "model_best.pth"))
            return False
        self.no_improve += 1
        return self.no_improve >= self.patience