"""
Encoder model with 5 layers of BiLSTM 
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    Encoder model with 5 layers of BiLSTM with hidden size 256
    """
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout, num_tags):
        super(Encoder, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
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


class EncoderForClassification(nn.Module):
    """
    Encoder model with BiLSTM for text classification
    For classification task (many-to-one) - outputs single label for entire sentence
    Uses Global Max Pooling to aggregate sequence features
    """
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout, num_classes):
        super(EncoderForClassification, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Bidirectional LSTM
        self.bilstm = nn.LSTM(
            embedding_dim, 
            hidden_size, 
            num_layers, 
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # BiLSTM output is 2 * hidden_size (forward + backward)
        self.fc = nn.Linear(2 * hidden_size, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Embedding layer
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        embedded = self.dropout(embedded)
        
        # BiLSTM layers
        lstm_out, (hidden, cell) = self.bilstm(embedded)
        # lstm_out shape: (batch_size, seq_length, 2 * hidden_size)
        
        # Permute to (batch, 2*hidden, seq_len) for max_pool1d
        lstm_out = lstm_out.permute(0, 2, 1)  # (batch_size, 2 * hidden_size, seq_length)
        
        # Global Max Pooling: compress seq_length dimension to 1
        pooled = F.max_pool1d(lstm_out, kernel_size=lstm_out.shape[2]).squeeze(2)
        # pooled shape: (batch_size, 2 * hidden_size)
        
        pooled = self.dropout(pooled)
        
        # Final classification layer
        output = self.fc(pooled)  # (batch_size, num_classes)
        
        return output