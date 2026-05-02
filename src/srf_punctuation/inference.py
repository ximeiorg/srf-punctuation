import json
from pathlib import Path
from typing import Dict, List

import torch

from .config import Config
from .models import PunctuationPredictor


class PunctuationInference:
    def __init__(
        self,
        model_path: str,
        vocab_path: str,
        config: Config = None,
    ):
        self.config = config or Config()
        self.vocab: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}
        self._load_vocab(vocab_path)
        self._load_model(model_path)

    def _load_vocab(self, vocab_path: str) -> None:
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.id_to_char = {v: k for k, v in self.vocab.items()}

    def _load_model(self, model_path: str) -> None:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        self.model = PunctuationPredictor(
            vocab_size=self.config.model.vocab_size,
            embed_dim=self.config.model.embed_dim,
            hidden_dim=self.config.model.hidden_dim,
            num_layers=self.config.model.num_layers,
            num_heads=self.config.model.num_heads,
            num_labels=self.config.model.num_labels,
            dropout=0.0,
            max_seq_len=self.config.model.max_seq_len,
        )
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            model_state = {}
            for k, v in state_dict.items():
                if k.startswith("model."):
                    model_state[k[6:]] = v
            self.model.load_state_dict(model_state)
        else:
            self.model.load_state_dict(checkpoint)
        self.model.eval()

    def text_to_ids(self, text: str) -> List[int]:
        return [self.vocab.get(char, self.vocab["<UNK>"]) for char in text]

    def predict(self, text: str) -> str:
        char_ids = self.text_to_ids(text)
        if len(char_ids) > self.config.model.max_seq_len:
            char_ids = char_ids[: self.config.model.max_seq_len]

        input_ids = torch.tensor([char_ids], dtype=torch.long)
        attention_mask = torch.ones(1, len(char_ids), dtype=torch.long)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)
            logits = outputs["logits"]
            preds = torch.argmax(logits, dim=-1)[0].tolist()

        punctuation_tokens = self.config.punctuation_tokens
        result = []
        for i, char in enumerate(text[: len(preds)]):
            result.append(char)
            label = preds[i]
            for name, idx in self.config.punctuation_map.items():
                if idx == label and name != "O":
                    result.append(punctuation_tokens[name])
                    break

        return "".join(result)

    def predict_batch(self, texts: List[str]) -> List[str]:
        return [self.predict(text) for text in texts]


def load_inference(
    checkpoint_path: str = "checkpoints/punctuation-epoch=01-val_loss=0.6772.ckpt",
    vocab_path: str = "data/vocab.json",
) -> PunctuationInference:
    return PunctuationInference(checkpoint_path, vocab_path)