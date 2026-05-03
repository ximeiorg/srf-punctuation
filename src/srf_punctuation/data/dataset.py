import json
import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset


class PunctuationDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        max_seq_len: int = 256,
        pad_id: int = 0,
        vocab: Dict[str, int] = None,
        enable_augmentation: bool = False,
        aug_keep_original: float = 0.3,
        aug_remove_punct: float = 0.4,
        aug_replace_punct: float = 0.2,
        aug_add_noise: float = 0.1,
        punctuation_tokens: Dict[str, str] = None,
        chinese_punctuation: List[str] = None,
    ):
        self.data_path = Path(data_path)
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id
        self.vocab = vocab or {}
        self.enable_augmentation = enable_augmentation
        self.aug_keep_original = aug_keep_original
        self.aug_remove_punct = aug_remove_punct
        self.aug_replace_punct = aug_replace_punct
        self.aug_add_noise = aug_add_noise
        self.punctuation_tokens = punctuation_tokens or {
            "O": "",
            "COMMA": "，",
            "PERIOD": "。",
            "QUESTION": "？",
            "EXCLAMATION": "！",
        }
        self.chinese_punctuation = chinese_punctuation or [
            "，", "。", "？", "！", "、", "；", "：", """, """, """, """, "（", "）", "【", "】", "《", "》"
        ]
        self.label_to_type = {0: "O", 1: "COMMA", 2: "PERIOD", 3: "QUESTION", 4: "EXCLAMATION"}
        self.line_offsets: List[int] = []
        self._build_index()

    def _build_index(self) -> None:
        import os
        index_path = self.data_path.with_suffix(".index")
        
        if index_path.exists():
            print(f"Loading cached index from {index_path}")
            import pickle
            with open(index_path, "rb") as f:
                self.line_offsets = pickle.load(f)
            print(f"Loaded {len(self.line_offsets)} samples")
            return
        
        print(f"Building index for {self.data_path}...")
        with open(self.data_path, "rb") as f:
            offset = 0
            for i, line in enumerate(f):
                if line.strip():
                    self.line_offsets.append(offset)
                offset += len(line)
                if (i + 1) % 100000 == 0:
                    print(f"  Indexed {i + 1:,} lines...")
        
        import pickle
        with open(index_path, "wb") as f:
            pickle.dump(self.line_offsets, f)
        print(f"Indexed {len(self.line_offsets)} samples, cached to {index_path}")

    def _augment_text(self, clean_text: str, labels: List[int]) -> str:
        if not self.enable_augmentation:
            return clean_text
        
        rand = random.random()
        keep_prob = self.aug_keep_original
        remove_prob = self.aug_remove_punct
        replace_prob = self.aug_replace_punct
        
        if rand < keep_prob:
            augmented = []
            for i, char in enumerate(clean_text):
                augmented.append(char)
                if labels[i] > 0:
                    punct = self.punctuation_tokens[self.label_to_type[labels[i]]]
                    augmented.append(punct)
            return "".join(augmented)
        
        elif rand < keep_prob + remove_prob:
            augmented = []
            for i, char in enumerate(clean_text):
                augmented.append(char)
                if labels[i] > 0 and random.random() > 0.5:
                    punct = self.punctuation_tokens[self.label_to_type[labels[i]]]
                    augmented.append(punct)
            return "".join(augmented)
        
        elif rand < keep_prob + remove_prob + replace_prob:
            augmented = []
            punct_types = ["COMMA", "PERIOD", "QUESTION", "EXCLAMATION"]
            for i, char in enumerate(clean_text):
                augmented.append(char)
                if labels[i] > 0:
                    if random.random() < 0.5:
                        correct_type = self.label_to_type[labels[i]]
                        wrong_types = [t for t in punct_types if t != correct_type]
                        wrong_type = random.choice(wrong_types)
                        punct = self.punctuation_tokens[wrong_type]
                    else:
                        punct = self.punctuation_tokens[self.label_to_type[labels[i]]]
                    augmented.append(punct)
            return "".join(augmented)
        
        else:
            augmented = []
            punct_types = ["COMMA", "PERIOD", "QUESTION", "EXCLAMATION"]
            for i, char in enumerate(clean_text):
                if random.random() < 0.1:
                    wrong_punct = self.punctuation_tokens[random.choice(punct_types)]
                    augmented.append(wrong_punct)
                augmented.append(char)
                if labels[i] > 0 and random.random() < 0.7:
                    punct = self.punctuation_tokens[self.label_to_type[labels[i]]]
                    augmented.append(punct)
            return "".join(augmented)

    def __len__(self) -> int:
        return len(self.line_offsets)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        with open(self.data_path, "rb") as f:
            f.seek(self.line_offsets[idx])
            line = f.readline().decode("utf-8")
            item = json.loads(line.strip())
        
        clean_text = item["text"]
        labels = item["labels"]
        
        augmented_text = self._augment_text(clean_text, labels)
        
        clean_chars = [c for c in augmented_text if c not in self.chinese_punctuation]
        char_ids = [self.vocab.get(c, self.vocab.get("<UNK>", 1)) for c in clean_chars]
        
        seq_len = min(len(char_ids), self.max_seq_len)
        char_ids = char_ids[:seq_len]
        labels = labels[:seq_len]
        
        padding_length = self.max_seq_len - len(char_ids)
        if padding_length > 0:
            char_ids = char_ids + [self.pad_id] * padding_length
            labels = labels + [-100] * padding_length
        
        attention_mask = [1] * seq_len + [0] * padding_length
        
        return {
            "input_ids": torch.tensor(char_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


class PunctuationDataModule:
    def __init__(
        self,
        data_dir: str,
        vocab: Dict[str, int] = None,
        max_seq_len: int = 256,
        batch_size: int = 32,
        num_workers: int = 4,
        pad_id: int = 0,
        enable_augmentation: bool = False,
        aug_keep_original: float = 0.3,
        aug_remove_punct: float = 0.4,
        aug_replace_punct: float = 0.2,
        aug_add_noise: float = 0.1,
        punctuation_tokens: Dict[str, str] = None,
        chinese_punctuation: List[str] = None,
    ):
        self.data_dir = Path(data_dir)
        self.vocab = vocab or {}
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pad_id = pad_id
        self.enable_augmentation = enable_augmentation
        self.aug_keep_original = aug_keep_original
        self.aug_remove_punct = aug_remove_punct
        self.aug_replace_punct = aug_replace_punct
        self.aug_add_noise = aug_add_noise
        self.punctuation_tokens = punctuation_tokens
        self.chinese_punctuation = chinese_punctuation

        self.train_dataset: Optional[PunctuationDataset] = None
        self.val_dataset: Optional[PunctuationDataset] = None
        self.test_dataset: Optional[PunctuationDataset] = None

    def setup(self) -> None:
        aug_params = {
            "vocab": self.vocab,
            "enable_augmentation": self.enable_augmentation,
            "aug_keep_original": self.aug_keep_original,
            "aug_remove_punct": self.aug_remove_punct,
            "aug_replace_punct": self.aug_replace_punct,
            "aug_add_noise": self.aug_add_noise,
            "punctuation_tokens": self.punctuation_tokens,
            "chinese_punctuation": self.chinese_punctuation,
        }
        
        self.train_dataset = PunctuationDataset(
            self.data_dir / "train.jsonl",
            max_seq_len=self.max_seq_len,
            pad_id=self.pad_id,
            **aug_params,
        )
        self.val_dataset = PunctuationDataset(
            self.data_dir / "val.jsonl",
            max_seq_len=self.max_seq_len,
            pad_id=self.pad_id,
            vocab=self.vocab,
            punctuation_tokens=self.punctuation_tokens,
            chinese_punctuation=self.chinese_punctuation,
        )
        self.test_dataset = PunctuationDataset(
            self.data_dir / "test.jsonl",
            max_seq_len=self.max_seq_len,
            pad_id=self.pad_id,
            vocab=self.vocab,
            punctuation_tokens=self.punctuation_tokens,
            chinese_punctuation=self.chinese_punctuation,
        )

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
        )