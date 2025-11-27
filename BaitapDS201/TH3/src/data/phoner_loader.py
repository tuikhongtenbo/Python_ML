"""
Data loader for PhoNER dataset (Named Entity Recognition)
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple
import os
import re

from data.preprocessing import Vocabulary


class PhoNERDataset(Dataset):
    """Dataset class for PhoNER"""
    
    def __init__(self, data_path: str, vocab: Vocabulary = None, tag2idx: Dict[str, int] = None,
                 max_length: int = 128):
        """
        Initialize PhoNER dataset
        """
        self.data_path = data_path
        self.max_length = max_length
        
        # Load data 
        self.sentences, self.tags = self._load_json(data_path)
        
        # Build vocabulary if not provided
        if vocab is None:
            self.vocab = Vocabulary()
            sentence_strings = [' '.join(sentence) for sentence in self.sentences]
            self.vocab.build_vocab(sentence_strings, min_freq=1)
        else:
            self.vocab = vocab
        
        # Create tag mapping if not provided
        if tag2idx is None:
            all_tags = []
            for tag_seq in self.tags:
                all_tags.extend(tag_seq)
            unique_tags = sorted(set(all_tags))
            self.tag2idx = {tag: idx for idx, tag in enumerate(unique_tags)}
        else:
            self.tag2idx = tag2idx
        
        self.idx2tag = {idx: tag for tag, idx in self.tag2idx.items()}
    
    def _load_json(self, file_path: str) -> Tuple[List[List[str]], List[List[str]]]:
        """
        Load data from JSON format 
        """
        sentences = []
        tags = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                item = json.loads(line)
                if 'words' in item and 'tags' in item:
                    sentences.append(item['words'])
                    tags.append(item['tags'])
        
        return sentences, tags
    
    def _tokenize(self, text: str) -> List[str]:
        return text.split()
    
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        tags = self.tags[idx]
        
        # Encode sentence - convert tokens to indices
        encoded = []
        for token in sentence:
            if token in self.vocab.word2idx:
                encoded.append(self.vocab.word2idx[token])
            else:
                encoded.append(self.vocab.UNK_IDX)
        
        # Pad or truncate to max_length
        if len(encoded) > self.max_length:
            encoded = encoded[:self.max_length]
        else:
            encoded = encoded + [self.vocab.PAD_IDX] * (self.max_length - len(encoded))
        
        # Encode tags
        tag_indices = [self.tag2idx.get(tag, 0) for tag in tags]
        
        # Pad or truncate tags to match sequence length
        if len(tag_indices) > self.max_length:
            tag_indices = tag_indices[:self.max_length]
        else:
            # Pad with -1 
            tag_indices = tag_indices + [-1] * (self.max_length - len(tag_indices))
        
        # Convert to tensors
        input_ids = torch.tensor(encoded, dtype=torch.long)
        tag_ids = torch.tensor(tag_indices, dtype=torch.long)
        
        return input_ids, tag_ids
    
    def get_num_tags(self):
        return len(self.tag2idx)


def create_ner_dataloaders(train_path: str, dev_path: str, test_path: str = None,
                           batch_size: int = 32, max_length: int = 128,
                           num_workers: int = 0):
    """
    Create train, dev, and test dataloaders for NER task
    """
    # Load training data to build vocabulary and tag mapping
    train_dataset = PhoNERDataset(train_path, vocab=None, tag2idx=None, max_length=max_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    vocab = train_dataset.vocab
    tag2idx = train_dataset.tag2idx
    
    # Create dev loader 
    dev_dataset = PhoNERDataset(dev_path, vocab=vocab, tag2idx=tag2idx, max_length=max_length)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Create test loader 
    test_dataset = PhoNERDataset(test_path, vocab=vocab, tag2idx=tag2idx, max_length=max_length)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, dev_loader, test_loader, vocab, tag2idx