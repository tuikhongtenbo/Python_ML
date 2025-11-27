"""
Configuration loader 
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Literal, Dict, Any
import yaml
import os
from pathlib import Path


class ModelConfig(BaseModel):
    """Model architecture configuration"""
    name: str = Field(..., description="Model name: LSTM, GRU, or Encoder")
    vocab_size: Optional[int] = Field(None, description="Vocabulary size (auto-determined)")
    embedding_dim: int = Field(128, gt=0, description="Embedding dimension")
    hidden_size: int = Field(256, gt=0, description="Hidden dimension")
    num_layers: int = Field(5, gt=0, description="Number of layers")
    dropout: float = Field(0.3, ge=0.0, le=1.0, description="Dropout rate")
    
    # For classification models (LSTM/GRU)
    num_classes: Optional[int] = Field(None, description="Number of classes (auto-determined)")
    
    # For NER models (Encoder)
    num_tags: Optional[int] = Field(None, description="Number of tags (auto-determined)")
    
    @validator('name')
    def validate_model_name(cls, v):
        """Validate model name"""
        valid_names = ['lstm', 'gru', 'encoder', 'LSTM', 'GRU', 'Encoder']
        if v.lower() not in [n.lower() for n in valid_names]:
            raise ValueError(f"Invalid model name: {v}. Must be one of: {valid_names}")
        return v


class DataConfig(BaseModel):
    """Data paths and configuration"""
    train_path: str = Field(..., description="Path to training data file")
    dev_path: str = Field(..., description="Path to dev/validation data file")
    test_path: Optional[str] = Field(None, description="Path to test data file")
    
    # For classification tasks
    task: Optional[str] = Field("sentiment", description="Task type: sentiment, topic, etc.")
    
    # Common settings
    max_length: int = Field(128, gt=0, description="Maximum sequence length")
    batch_size: int = Field(32, gt=0, description="Batch size")
    num_workers: int = Field(0, ge=0, description="Number of data loader workers")


class TrainingConfig(BaseModel):
    """Training hyperparameters"""
    num_epochs: int = Field(20, gt=0, description="Number of epochs")
    learning_rate: float = Field(0.001, gt=0.0, description="Learning rate")
    device: Literal["cuda", "cpu"] = Field("cuda", description="Device: cuda or cpu")


class PathsConfig(BaseModel):
    """Paths for logs and model saving"""
    log_dir: str = Field("logs", description="Directory for logs")
    model_save_dir: str = Field("models", description="Directory for saving models")
    model_name: str = Field(..., description="Model filename")


class Config(BaseModel):
    """
    Main configuration class for NER and Classification models.
    """
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    paths: PathsConfig
    
    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> 'Config':
        """
        Load configuration from YAML file with Pydantic validation.
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        return cls(**config_data)
    
    def save_yaml(self, yaml_path: str | Path):
        """
        Save configuration to YAML file.
        """
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(
                self.model_dump(),
                f,
                default_flow_style=False,
                allow_unicode=True,
                sort_keys=False
            )
    
    def get(self, key: str, default=None):
        """
        Get nested configuration value using dot notation.
        """
        keys = key.split('.')
        value = self
        
        try:
            for k in keys:
                if hasattr(value, k):
                    value = getattr(value, k)
                else:
                    return default
            return value
        except (AttributeError, KeyError):
            return default
    
    class Config:
        """Pydantic config."""
        extra = "allow"  # Allow extra fields
        validate_assignment = True  # Validate on assignment


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file 
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Config object
    """
    return Config.from_yaml(config_path)


def get_config_path(model_name: str) -> str:
    """
    Get config path for a model
    
    Args:
        model_name: Name of the model (e.g., 'lstm', 'gru')
        
    Returns:
        Path to config file (absolute path)
    """
    # Get the directory where this file is located (src/config/)
    config_dir = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(config_dir, f"{model_name}.yaml")
 
    if os.path.exists(config_file):
        return config_file
    
    # Fallback: try relative path from current working directory
    relative_path = os.path.join("src", "config", f"{model_name}.yaml")
    if os.path.exists(relative_path):
        return relative_path
    
    return relative_path