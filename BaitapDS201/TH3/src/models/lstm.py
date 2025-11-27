"""
LSTM model with 5 layers and hidden size is 256
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    """
    LSTM model with 5 layers and hidden size is 256
    """
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout, num_classes):
        
        super(LSTM, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        
        # Embedding layer
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        
        # LSTM layer
        lstm_out, (hidden, cell) = self.lstm(embedded)  # lstm_out: (batch_size, seq_length, hidden_size)
        
        lstm_out = lstm_out.permute(0, 2, 1)  # (batch_size, hidden_size, seq_length)
        
        # Global Max Pooling: compress seq_length dimension to 1
        pooled = F.max_pool1d(lstm_out, kernel_size=lstm_out.shape[2]).squeeze(2)  # (batch_size, hidden_size)
        
        # Fully connected layers
        out = self.fc1(pooled)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        # Final classification layer
        out = self.fc3(out)  # (batch_size, num_classes)
        
        return out