import torch
import torch.nn as nn

class CNN_LSTM_Attention_Model(nn.Module):
    """
    Q1 Novelty: CNN extracts spatial features, LSTM models temporal sequence, 
    and Self-Attention dynamically weights the most critical time steps.
    """
    def __init__(self, hidden_size=256, num_layers=2, dropout=0.5):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(dropout)
        
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        
        # Self-Attention Layer
        self.attention_weights = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # CNN
        x = x.permute(0, 2, 1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        
        # LSTM
        lstm_out, _ = self.lstm(x) # (batch, seq_len, hidden_size)
        
        # Self-Attention
        attn_scores = self.attention_weights(lstm_out) # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)
        
        # Context vector
        context = torch.sum(attn_weights * lstm_out, dim=1) # (batch, hidden_size)
        context = self.dropout(context)
        
        return self.fc(context)