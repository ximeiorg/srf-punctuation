from dataclasses import dataclass, field
from typing import List


@dataclass
class ModelConfig:
    vocab_size: int = 8000
    embed_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    max_seq_len: int = 256
    num_labels: int = 5


@dataclass
class DataConfig:
    raw_data_dir: str = "rawdata"
    processed_data_dir: str = "data"
    train_file: str = "train.jsonl"
    val_file: str = "val.jsonl"
    test_file: str = "test.jsonl"
    vocab_file: str = "vocab.json"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    max_samples: int|None = None


@dataclass
class TrainingConfig:
    batch_size: int = 512
    learning_rate: float = 1e-3
    max_epochs: int = 20
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_clip_val: float = 1.0
    early_stopping_patience: int = 5
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    punctuation_map: dict = field(
        default_factory=lambda: {
            "O": 0,
            "COMMA": 1,
            "PERIOD": 2,
            "QUESTION": 3,
            "EXCLAMATION": 4,
        }
    )

    punctuation_tokens: dict = field(
        default_factory=lambda: {
            "O": "",
            "COMMA": "，",
            "PERIOD": "。",
            "QUESTION": "？",
            "EXCLAMATION": "！",
        }
    )

    chinese_punctuation: List[str] = field(
        default_factory=lambda: ["，", "。", "？", "！", "、", "；", "：", """, """, """, """, "（", "）", "【", "】", "《", "》"]
    )