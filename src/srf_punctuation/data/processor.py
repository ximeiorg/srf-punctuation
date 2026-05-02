import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from ..config import Config

console = Console()


class DataProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.vocab: Dict[str, int] = {}
        self.id_to_char: Dict[int, str] = {}

    def load_raw_data(self) -> List[str]:
        texts = []
        raw_dir = Path(self.config.data.raw_data_dir)

        jsonl_file = raw_dir / "distill_r1_110k.jsonl"
        if jsonl_file.exists():
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn(), console=console) as progress:
                with open(jsonl_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                task = progress.add_task("[cyan]Processing distill_r1_110k.jsonl", total=len(lines))
                for line in lines:
                    if self.config.data.max_samples is not None and len(texts) >= self.config.data.max_samples:
                        break
                    try:
                        data = json.loads(line.strip())
                        if "content" in data and data["content"]:
                            texts.append(data["content"])
                    except json.JSONDecodeError:
                        pass
                    progress.update(task, advance=1)

        qwen_file = raw_dir / "qwen3_235b_2507_distill_110k.jsonl"
        if qwen_file.exists() and (self.config.data.max_samples is None or len(texts) < self.config.data.max_samples):
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn(), console=console) as progress:
                with open(qwen_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                task = progress.add_task("[green]Processing qwen3_235b_2507_distill_110k.jsonl", total=len(lines))
                for line in lines:
                    if self.config.data.max_samples is not None and len(texts) >= self.config.data.max_samples:
                        break
                    try:
                        data = json.loads(line.strip())
                        if "messages" in data:
                            for msg in data["messages"]:
                                if "content" in msg and msg["content"] and len(msg["content"]) > 20:
                                    texts.append(msg["content"])
                    except json.JSONDecodeError:
                        pass
                    progress.update(task, advance=1)

        csv_file = raw_dir / "ChnSentiCorp_htl_all.csv"
        if csv_file.exists() and (self.config.data.max_samples is None or len(texts) < self.config.data.max_samples):
            with open(csv_file, "r", encoding="utf-8") as f:
                lines = f.readlines()[1:]
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn(), console=console) as progress:
                task = progress.add_task("[yellow]Processing ChnSentiCorp_htl_all.csv", total=len(lines))
                for line in lines:
                    if self.config.data.max_samples is not None and len(texts) >= self.config.data.max_samples:
                        break
                    parts = line.strip().split(",", 1)
                    if len(parts) == 2 and len(parts[1]) > 10:
                        texts.append(parts[1])
                    progress.update(task, advance=1)

        items_file = raw_dir / "items.json"
        if items_file.exists() and (self.config.data.max_samples is None or len(texts) < self.config.data.max_samples):
            with open(items_file, "r", encoding="utf-8") as f:
                items = json.load(f)
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn(), console=console) as progress:
                task = progress.add_task("[magenta]Processing items.json", total=len(items))
                for item in items:
                    if self.config.data.max_samples is not None and len(texts) >= self.config.data.max_samples:
                        break
                    if "headline" in item and item["headline"] and len(item["headline"]) > 10:
                        texts.append(item["headline"])
                    if "description" in item and item["description"] and len(item["description"]) > 20:
                        texts.append(item["description"])
                    progress.update(task, advance=1)

        console.print(f"[bold green]Loaded {len(texts)} raw texts[/bold green]")
        return texts

    def extract_punctuation_labels(
        self, text: str
    ) -> Tuple[str, List[int]]:
        chars = list(text)
        clean_chars = []
        labels = []

        punctuation_map = {
            "，": 1,
            "。": 2,
            "？": 3,
            "！": 4,
            "、": 1,
            "；": 1,
            "：": 1,
        }

        i = 0
        while i < len(chars):
            char = chars[i]
            if char in self.config.chinese_punctuation:
                if clean_chars and clean_chars[-1] not in self.config.chinese_punctuation:
                    pun_label = punctuation_map.get(char, 0)
                    labels[-1] = pun_label
            elif char.strip():
                clean_chars.append(char)
                labels.append(0)
            i += 1

        return "".join(clean_chars), labels

    def build_vocab(self, texts: List[str]) -> None:
        char_counter = Counter()
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn(), console=console) as progress:
            task = progress.add_task("[blue]Building vocabulary", total=len(texts))
            for text in texts:
                clean_text, _ = self.extract_punctuation_labels(text)
                char_counter.update(clean_text)
                progress.update(task, advance=1)

        special_tokens = ["<PAD>", "<UNK>", "<CLS>", "<SEP>"]
        self.vocab = {token: i for i, token in enumerate(special_tokens)}

        most_common = char_counter.most_common(self.config.model.vocab_size - len(special_tokens))
        for char, _ in most_common:
            if char not in self.vocab:
                self.vocab[char] = len(self.vocab)

        self.id_to_char = {v: k for k, v in self.vocab.items()}
        console.print(f"[bold cyan]Vocabulary size: {len(self.vocab)}[/bold cyan]")

    def save_vocab(self, output_dir: Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        vocab_path = output_dir / self.config.data.vocab_file
        with open(vocab_path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        print(f"Vocabulary saved to {vocab_path}")

    def load_vocab(self, vocab_path: Path) -> None:
        with open(vocab_path, "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.id_to_char = {v: k for k, v in self.vocab.items()}
        print(f"Vocabulary loaded: {len(self.vocab)} chars")

    def text_to_ids(self, text: str) -> List[int]:
        return [self.vocab.get(char, self.vocab["<UNK>"]) for char in text]

    def process_and_save(self) -> None:
        texts = self.load_raw_data()

        self.build_vocab(texts)

        processed_data = []
        max_len = self.config.model.max_seq_len
        min_len = 10
        label_counter = Counter()

        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TaskProgressColumn(), console=console) as progress:
            task = progress.add_task("[green]Processing texts into samples", total=len(texts))
            for text in texts:
                clean_text, labels = self.extract_punctuation_labels(text)
                label_counter.update(labels)
                
                if len(clean_text) <= max_len and len(clean_text) >= min_len:
                    char_ids = self.text_to_ids(clean_text)
                    processed_data.append(
                        {"text": clean_text, "char_ids": char_ids, "labels": labels}
                    )
                elif len(clean_text) > max_len:
                    for i in range(0, len(clean_text) - min_len, max_len // 2):
                        chunk_text = clean_text[i:i + max_len]
                        chunk_labels = labels[i:i + max_len]
                        if len(chunk_text) >= min_len:
                            char_ids = self.text_to_ids(chunk_text)
                            processed_data.append(
                                {"text": chunk_text, "char_ids": char_ids, "labels": chunk_labels}
                            )
                
                progress.update(task, advance=1)

        console.print(f"[bold green]Total samples: {len(processed_data)}[/bold green]")
        
        console.print("\n[bold yellow]Label distribution:[/bold yellow]")
        label_names = ["O(无标点)", "COMMA(逗号)", "PERIOD(句号)", "QUESTION(问号)", "EXCLAMATION(感叹号)"]
        total_labels = sum(label_counter.values())
        for label_id in range(5):
            count = label_counter.get(label_id, 0)
            pct = count / total_labels * 100 if total_labels > 0 else 0
            console.print(f"  {label_names[label_id]}: {count:,} ({pct:.2f}%)")

        total = len(processed_data)
        train_size = int(total * self.config.data.train_ratio)
        val_size = int(total * self.config.data.val_ratio)

        train_data = processed_data[:train_size]
        val_data = processed_data[train_size : train_size + val_size]
        test_data = processed_data[train_size + val_size :]

        output_dir = Path(self.config.data.processed_data_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._save_jsonl(train_data, output_dir / self.config.data.train_file)
        self._save_jsonl(val_data, output_dir / self.config.data.val_file)
        self._save_jsonl(test_data, output_dir / self.config.data.test_file)
        self.save_vocab(output_dir)

        console.print(f"[bold cyan]Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}[/bold cyan]")

    def _save_jsonl(self, data: List[dict], path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        console.print(f"[bold]Saved {len(data)} samples to {path}[/bold]")