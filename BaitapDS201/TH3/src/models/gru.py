"""
GRU model with 5 layers and hidden size is 256
"""

import torch
import torch.nn as nn


class GRU(nn.Module):
    """
    GRU model with 5 layers and hidden size is 256
    """
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout):
        
        super(GRU, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(hidden_size, 3)
    
    def forward(self, x):

        # Embedding layer
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        
        # GRU layer
        gru_out, hidden = self.gru(embedded)  # gru_out: (batch_size, seq_length, hidden_size)
        
        # Use the last output of the sequence
        last_output = gru_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Fully connected layers
        out = self.fc1(last_output)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        # Final classification layer
        out = self.fc3(out)  # (batch_size, 3)
        
        return out