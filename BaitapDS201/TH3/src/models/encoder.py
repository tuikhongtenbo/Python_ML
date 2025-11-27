"""
Encoder model with 5 layers of BiLSTM 
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Encoder model with 5 layers of BiLSTM with hidden size 256
    """
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout, num_tags):
        super(Encoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Bidirectional LSTM
        self.bilstm = nn.LSTM(
            embedding_dim, 
            hidden_size, 
            num_layers, 
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(2 * hidden_size, num_tags)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):

        # Embedding layer
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        embedded = self.dropout(embedded)
        
        # BiLSTM layers
        lstm_out, (hidden, cell) = self.bilstm(embedded)  
        # lstm_out: (batch_size, seq_length, 2 * hidden_size)
        
        # Apply dropout
        lstm_out = self.dropout(lstm_out)
        
        # Output layer for each token
        output = self.fc(lstm_out)  # (batch_size, seq_length, num_tags)
        
        return output