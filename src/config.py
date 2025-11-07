import yaml
from dataclasses import dataclass
from typing import Optional

@dataclass
class DataConfig:
    dataset_name: str = "iwslt2017"
    dataset_config: str = "iwslt2017-de-en"
    src_lang: str = "en"
    tgt_lang: str = "de"
    vocab_size: int = 8000
    max_seq_len: int = 100
    batch_size: int = 32
    train_split: str = "train"
    val_split: str = "validation"
    test_split: str = "test"

@dataclass
class ModelConfig:
    d_model: int = 512
    num_heads: int = 8
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    d_ff: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 100

@dataclass
class TrainingConfig:
    batch_size: int = 32
    max_epochs: int = 30
    base_learning_rate: float = 1.0e-4
    warmup_steps: int = 4000
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_epsilon: float = 1.0e-9
    grad_clip: float = 1.0
    label_smoothing: float = 0.1
    log_interval: int = 50
    save_interval: int = 5

class Config:
    """配置类"""
    
    def __init__(self, data: DataConfig, model: ModelConfig, training: TrainingConfig, 
                 device: str = "cpu", seed: int = 42):
        self.data = data
        self.model = model
        self.training = training
        self.device = device
        self.seed = seed

    @classmethod
    def from_yaml(cls, yaml_path: str):
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # 创建嵌套配置对象
        data_config = DataConfig(**data.get('data', {}))
        model_config = ModelConfig(**data.get('model', {}))
        training_config = TrainingConfig(**data.get('training', {}))
        
        return cls(
            data=data_config,
            model=model_config,
            training=training_config,
            device=data.get('device', 'cpu'),
            seed=data.get('seed', 42)
        )
    
    def to_dict(self):
        """将配置转换为字典，用于保存"""
        return {
            'data': {
                'dataset_name': self.data.dataset_name,
                'dataset_config': self.data.dataset_config,
                'src_lang': self.data.src_lang,
                'tgt_lang': self.data.tgt_lang,
                'vocab_size': self.data.vocab_size,
                'max_seq_len': self.data.max_seq_len,
                'batch_size': self.data.batch_size,
                'train_split': self.data.train_split,
                'val_split': self.data.val_split,
                'test_split': self.data.test_split,
            },
            'model': {
                'd_model': self.model.d_model,
                'num_heads': self.model.num_heads,
                'num_encoder_layers': self.model.num_encoder_layers,
                'num_decoder_layers': self.model.num_decoder_layers,
                'd_ff': self.model.d_ff,
                'dropout': self.model.dropout,
                'max_seq_len': self.model.max_seq_len,
            },
            'training': {
                'batch_size': self.training.batch_size,
                'max_epochs': self.training.max_epochs,
                'base_learning_rate': self.training.base_learning_rate,
                'warmup_steps': self.training.warmup_steps,
                'adam_beta1': self.training.adam_beta1,
                'adam_beta2': self.training.adam_beta2,
                'adam_epsilon': self.training.adam_epsilon,
                'grad_clip': self.training.grad_clip,
                'label_smoothing': self.training.label_smoothing,
                'log_interval': self.training.log_interval,
                'save_interval': self.training.save_interval,
            },
            'device': self.device,
            'seed': self.seed
        }