"""
Trainer for NER (Named Entity Recognition) models
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, classification_report
import numpy as np
from tqdm import tqdm
from typing import Dict

from utils.logger import setup_logger


class NERTrainer:
    """Trainer class for NER models"""
    
    def __init__(self, model: nn.Module, train_loader: DataLoader, dev_loader: DataLoader,
                 test_loader: DataLoader = None, device: str = 'cuda',
                 learning_rate: float = 0.001, log_dir: str = "logs"):
        """
        Initialize NER trainer
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.test_loader = test_loader
        self.device = device
        self.logger = setup_logger(log_dir, "ner_trainer")
        
        # Loss function 
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        
        # Optimizer (Adam)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training history
        self.history = {
            'train_loss': [],
            'dev_loss': [],
            'dev_f1': [],
            'dev_accuracy': []
        }
    
    def train_epoch(self) -> float:
        """
        Train for one epoch
        """
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (inputs, labels) in enumerate(progress_bar):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)  # (batch_size, seq_length, num_tags)
            
            # Reshape for loss calculation
            # Flatten sequence and tag dimensions
            outputs_flat = outputs.view(-1, outputs.size(-1))  # (batch_size * seq_length, num_tags)
            labels_flat = labels.view(-1)  # (batch_size * seq_length)
            
            loss = self.criterion(outputs_flat, labels_flat)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate average loss
        avg_loss = total_loss / len(self.train_loader)
        
        return avg_loss
    
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on the dataset
        """
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(data_loader, desc="Evaluating"):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)  # (batch_size, seq_length, num_tags)
                
                # Reshape for loss calculation
                outputs_flat = outputs.view(-1, outputs.size(-1))
                labels_flat = labels.view(-1)
                
                loss = self.criterion(outputs_flat, labels_flat)
                
                # Statistics
                total_loss += loss.item()
                predictions = torch.argmax(outputs, dim=2)
                
                # Flatten for metric calculation
                predictions_flat = predictions.view(-1).cpu().numpy()
                labels_flat_np = labels_flat.cpu().numpy()
                
                # Filter out padding
                mask = labels_flat_np != 0
                if mask.sum() > 0:
                    all_predictions.extend(predictions_flat[mask])
                    all_labels.extend(labels_flat_np[mask])
        
        # Calculate metrics
        avg_loss = total_loss / len(data_loader)
        if len(all_predictions) > 0:

            f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
            accuracy = accuracy_score(all_labels, all_predictions)
        else:
            f1 = 0.0
            accuracy = 0.0
        
        return {
            'loss': avg_loss,
            'f1': f1,
            'accuracy': accuracy,
            'predictions': all_predictions,
            'labels': all_labels
        }
    
    def train(self, num_epochs: int, save_path: str = None):
        """
        Train the model
        
        Args:
            num_epochs: Number of epochs to train
            save_path: Path to save the best model
        """
        best_f1 = 0.0
        
        self.logger.info(f"Starting training for {num_epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(1, num_epochs + 1):
            self.logger.info(f"\nEpoch {epoch}/{num_epochs}")
            
            # Train
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            
            # Evaluate on dev set
            dev_metrics = self.evaluate(self.dev_loader)
            self.history['dev_loss'].append(dev_metrics['loss'])
            self.history['dev_f1'].append(dev_metrics['f1'])
            self.history['dev_accuracy'].append(dev_metrics['accuracy'])
            
            # Log metrics
            self.logger.info(f"Train Loss: {train_loss:.4f}")
            self.logger.info(f"Dev Loss: {dev_metrics['loss']:.4f}, Dev F1: {dev_metrics['f1']:.4f}, "
                           f"Dev Accuracy: {dev_metrics['accuracy']:.4f}")
            
            # Save best model
            if dev_metrics['f1'] > best_f1:
                best_f1 = dev_metrics['f1']
                if save_path:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'f1': best_f1,
                        'history': self.history
                    }, save_path)
                    self.logger.info(f"Saved best model with F1: {best_f1:.4f}")
        
        self.logger.info(f"\nTraining completed. Best F1: {best_f1:.4f}")
    
    def test(self, model_path: str = None) -> Dict[str, float]:
        """
        Evaluate on test set
        
        Args:
            model_path: Path to saved model (if None, use current model)
            
        Returns:
            Dictionary with test metrics
        """
        if model_path:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.logger.info(f"Loaded model from {model_path}")
        
        if self.test_loader is None:
            self.logger.warning("No test loader provided")
            return {}
        
        self.logger.info("Evaluating on test set...")
        test_metrics = self.evaluate(self.test_loader)
        
        self.logger.info(f"Test Loss: {test_metrics['loss']:.4f}")
        self.logger.info(f"Test F1: {test_metrics['f1']:.4f}")
        self.logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        
        # Print classification report
        if hasattr(self, 'idx2tag'):
            target_names = [self.idx2tag[i] for i in sorted(self.idx2tag.keys())]
            report = classification_report(
                test_metrics['labels'],
                test_metrics['predictions'],
                target_names=target_names
            )
            self.logger.info(f"\nClassification Report:\n{report}")
        
        return test_metrics