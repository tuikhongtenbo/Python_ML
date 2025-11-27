"""
Data loader for UIT-VSFC dataset
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple
import os

from data.preprocessing import Vocabulary, create_label_mapping


class VSFCDataset(Dataset):    
    def __init__(self, data_path: str, vocab: Vocabulary = None, label2idx: Dict[str, int] = None, 
                 max_length: int = 128, task: str = 'sentiment'):
        self.data_path = data_path
        self.max_length = max_length
        self.task = task
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Extract sentences and labels
        self.sentences = [item['sentence'] for item in self.data]
        self.labels = [item[task] for item in self.data]
        
        # Build vocabulary 
        if vocab is None:
            self.vocab = Vocabulary()
            self.vocab.build_vocab(self.sentences, min_freq=1)
        else:
            self.vocab = vocab
        
        # Create label mapping 
        if label2idx is None:
            self.label2idx = create_label_mapping(self.labels)
        else:
            self.label2idx = label2idx
        
        self.idx2label = {idx: label for label, idx in self.label2idx.items()}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        encoded = self.vocab.encode(sentence, max_length=self.max_length)
        input_ids = torch.tensor(encoded, dtype=torch.long)
        label_idx = torch.tensor(self.label2idx[label], dtype=torch.long)
        
        return input_ids, label_idx
    
    def get_num_classes(self):
        return len(self.label2idx)


def create_dataloaders(train_path: str, dev_path: str, test_path: str = None,
                      batch_size: int = 32, max_length: int = 128, 
                      task: str = 'sentiment', num_workers: int = 0):

    # Load training data to build vocabulary
    train_dataset = VSFCDataset(train_path, vocab=None, label2idx=None, max_length=max_length, task=task)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    vocab = train_dataset.vocab
    label2idx = train_dataset.label2idx

    # Create dev loader
    dev_dataset = VSFCDataset(dev_path, vocab=vocab, label2idx=label2idx, max_length=max_length, task=task)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Create test loader 
    test_dataset = VSFCDataset(test_path, vocab=vocab, label2idx=label2idx, max_length=max_length, task=task)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, dev_loader, test_loader, vocab, label2idx