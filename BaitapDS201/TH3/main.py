"""
Main script for training models:
- LSTM/GRU for text classification on UIT-VSFC dataset
- Encoder (BiLSTM) for Named Entity Recognition on PhoNER dataset
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# Add src to path
src_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
sys.path.insert(0, src_path)

from config import load_config, get_config_path
from data import create_dataloaders, create_ner_dataloaders
from models import LSTM, GRU, Encoder
from trainer_vsfc import Trainer
from trainer_ner import NERTrainer


def main():
    parser = argparse.ArgumentParser(description='Train models (LSTM/GRU for classification or Encoder for NER)')
    parser.add_argument('--config', type=str, default='lstm.yaml')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--model_path', type=str, default=None)
    args = parser.parse_args()
    
    # Load configuration
    config_path = get_config_path(args.config.replace('.yaml', ''))
    config = load_config(config_path)
    
    # Set device
    device = config.training.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    # Determine model type and task type
    model_name = config.model.name.lower()
    is_ner_task = model_name == 'encoder'
    
    # Create data loaders
    print("Loading data...")
    if is_ner_task:
        # NER task
        train_loader, dev_loader, test_loader, vocab, tag2idx = create_ner_dataloaders(
            train_path=config.data.train_path,
            dev_path=config.data.dev_path,
            test_path=config.data.test_path,
            batch_size=config.data.batch_size,
            max_length=config.data.max_length,
            num_workers=config.data.num_workers
        )
        
        print(f"Vocabulary size: {len(vocab)}")
        print(f"Number of tags: {len(tag2idx)}")
        print(f"Tag mapping: {tag2idx}")
        
        # Create Encoder model
        model = Encoder(
            vocab_size=len(vocab),
            embedding_dim=config.model.embedding_dim,
            hidden_size=config.model.hidden_size,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout,
            num_tags=len(tag2idx)
        )
        
        print(f"\nModel architecture:")
        print(f"  Model: Encoder (BiLSTM)")
        print(f"  Embedding dim: {config.model.embedding_dim}")
        print(f"  Hidden size: {config.model.hidden_size}")
        print(f"  Number of BiLSTM layers: {config.model.num_layers}")
        print(f"  Dropout: {config.model.dropout}")
        print(f"  Number of tags: {len(tag2idx)}")
        print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Create NER trainer
        trainer = NERTrainer(
            model=model,
            train_loader=train_loader,
            dev_loader=dev_loader,
            test_loader=test_loader,
            device=device,
            learning_rate=config.training.learning_rate,
            log_dir=config.paths.log_dir
        )
        
        # Store tag mapping for classification report
        trainer.idx2tag = {idx: tag for tag, idx in tag2idx.items()}
        
    else:
        # Text classification task (LSTM or GRU)
        train_loader, dev_loader, test_loader, vocab, label2idx = create_dataloaders(
            train_path=config.data.train_path,
            dev_path=config.data.dev_path,
            test_path=config.data.test_path,
            batch_size=config.data.batch_size,
            max_length=config.data.max_length,
            task=config.data.task,
            num_workers=config.data.num_workers
        )
        
        print(f"Vocabulary size: {len(vocab)}")
        print(f"Number of classes: {len(label2idx)}")
        print(f"Label mapping: {label2idx}")
        
        # Create model (LSTM or GRU)
        if model_name == 'gru':
            model = GRU(
                vocab_size=len(vocab),
                embedding_dim=config.model.embedding_dim,
                hidden_size=config.model.hidden_size,
                num_layers=config.model.num_layers,
                dropout=config.model.dropout
            )
        else:  # Default to LSTM
            model = LSTM(
                vocab_size=len(vocab),
                embedding_dim=config.model.embedding_dim,
                hidden_size=config.model.hidden_size,
                num_layers=config.model.num_layers,
                dropout=config.model.dropout
            )
        
        print(f"\nModel architecture:")
        print(f"  Model: {model_name.upper()}")
        print(f"  Embedding dim: {config.model.embedding_dim}")
        print(f"  Hidden size: {config.model.hidden_size}")
        print(f"  Number of layers: {config.model.num_layers}")
        print(f"  Dropout: {config.model.dropout}")
        print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            dev_loader=dev_loader,
            test_loader=test_loader,
            device=device,
            learning_rate=config.training.learning_rate,
            log_dir=config.paths.log_dir
        )
        
        # Store label mapping for classification report
        trainer.idx2label = {idx: label for label, idx in label2idx.items()}
    
    if args.mode == 'train':
        # Create model save directory
        os.makedirs(config.paths.model_save_dir, exist_ok=True)
        model_save_path = os.path.join(config.paths.model_save_dir, 
                                      config.paths.model_name)
        
        # Train model
        trainer.train(
            num_epochs=config.training.num_epochs,
            save_path=model_save_path
        )
        
        # Test on test set
        print("\n" + "="*50)
        print("Evaluating on test set...")
        test_metrics = trainer.test(model_path=model_save_path)
        
    elif args.mode == 'test':
        if args.model_path is None:
            model_path = os.path.join(config.paths.model_save_dir,
                                    config.paths.model_name)
        else:
            model_path = args.model_path
        
        if not os.path.exists(model_path):
            print(f"Error: Model file not found at {model_path}")
            sys.exit(1)
        
        # Test model
        test_metrics = trainer.test(model_path=model_path)


if __name__ == "__main__":
    main()