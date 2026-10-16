import torch
from torch.utils.data import DataLoader, TensorDataset
from core.model import CNN_LSTM_Attention_Model

class FederatedNode:
    def __init__(self, X, y, device, batch_size=128):
        self.device = device
        self.model = CNN_LSTM_Attention_Model().to(device)
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        self.loader = DataLoader(
            TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=True)
        
        pos_count = float(y.sum())
        neg_count = float(len(y) - pos_count)
        if pos_count > 0:
            self.pos_weight = torch.tensor([neg_count / pos_count], device=device)
        else:
            self.pos_weight = torch.tensor([1.0], device=device)

    def load_full_state(self, state):
        self.model.load_state_dict(state)

    def train_local(self, epochs=5, lr=0.0005, weight_decay=1e-5):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self.model.train()
        total_loss = 0.0
        for _ in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in self.loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                optimizer.zero_grad()
                loss = loss_fn(self.model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            total_loss += epoch_loss / max(len(self.loader), 1)
        return {
            "weights": {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
            "loss": total_loss / epochs,
        }