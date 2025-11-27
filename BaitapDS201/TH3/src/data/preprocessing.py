"""
Text preprocessing utilities 
"""

import re
from collections import Counter
from typing import List, Dict


class Vocabulary:
    """Vocabulary class for text tokenization"""
    
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.word_count = Counter()
        
        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.PAD_IDX = 0
        self.UNK_IDX = 1
        
        # Initialize with special tokens
        self.word2idx[self.PAD_TOKEN] = self.PAD_IDX
        self.word2idx[self.UNK_TOKEN] = self.UNK_IDX
        self.idx2word[self.PAD_IDX] = self.PAD_TOKEN
        self.idx2word[self.UNK_IDX] = self.UNK_TOKEN
        
    def build_vocab(self, sentences: List[str], min_freq: int = 1):
        """
        Build vocabulary from sentences
        """
        # Count word frequencies
        for sentence in sentences:
            words = self.tokenize(sentence)
            self.word_count.update(words)
        
        # Add words to vocabulary
        idx = len(self.word2idx)
        for word, count in self.word_count.items():
            if count >= min_freq and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize Vietnamese text
        """
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        
        return tokens
    
    def encode(self, sentence: str, max_length: int = None) -> List[int]:
        """
        Encode a sentence to indices
        """
        tokens = self.tokenize(sentence)
        indices = [self.word2idx.get(token, self.UNK_IDX) for token in tokens]
        
        if max_length:
            if len(indices) > max_length:
                indices = indices[:max_length]
            else:
                indices = indices + [self.PAD_IDX] * (max_length - len(indices))
        
        return indices
    
    def __len__(self):
        return len(self.word2idx)


def create_label_mapping(labels: List[str]) -> Dict[str, int]:
    """
    Create label to index mapping
    """
    unique_labels = sorted(set(labels))
    label2idx = {label: idx for idx, label in enumerate(unique_labels)}
    return label2idx