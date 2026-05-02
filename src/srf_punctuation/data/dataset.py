import json
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
    ):
        self.data_path = Path(data_path)
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id
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

    def __len__(self) -> int:
        return len(self.line_offsets)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        with open(self.data_path, "rb") as f:
            f.seek(self.line_offsets[idx])
            line = f.readline().decode("utf-8")
            item = json.loads(line.strip())
        char_ids = item["char_ids"]
        labels = item["labels"]

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
        max_seq_len: int = 256,
        batch_size: int = 32,
        num_workers: int = 4,
        pad_id: int = 0,
    ):
        self.data_dir = Path(data_dir)
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pad_id = pad_id

        self.train_dataset: Optional[PunctuationDataset] = None
        self.val_dataset: Optional[PunctuationDataset] = None
        self.test_dataset: Optional[PunctuationDataset] = None

    def setup(self) -> None:
        self.train_dataset = PunctuationDataset(
            self.data_dir / "train.jsonl",
            max_seq_len=self.max_seq_len,
            pad_id=self.pad_id,
        )
        self.val_dataset = PunctuationDataset(
            self.data_dir / "val.jsonl",
            max_seq_len=self.max_seq_len,
            pad_id=self.pad_id,
        )
        self.test_dataset = PunctuationDataset(
            self.data_dir / "test.jsonl",
            max_seq_len=self.max_seq_len,
            pad_id=self.pad_id,
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