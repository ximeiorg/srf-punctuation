import json
from pathlib import Path
from collections import Counter

import torch


def compute_class_weights(data_dir: str = "data", max_samples: int = None):
    data_path = Path(data_dir)
    train_file = data_path / "train.jsonl"
    
    print(f"Computing class weights from {train_file}")
    
    label_counts = Counter()
    total_labels = 0
    
    with open(train_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            if not line.strip():
                continue
            item = json.loads(line.strip())
            labels = item["labels"]
            label_counts.update(labels)
            total_labels += len(labels)
            
            if (i + 1) % 100000 == 0:
                print(f"  Processed {i + 1:,} lines...")
    
    print(f"\nLabel distribution:")
    num_classes = 5
    class_names = ["O(无标点)", "COMMA(逗号)", "PERIOD(句号)", "QUESTION(问号)", "EXCLAMATION(感叹号)"]
    
    for cls in range(num_classes):
        count = label_counts.get(cls, 0)
        ratio = count / total_labels * 100
        print(f"  Class {cls} ({class_names[cls]}): {count:,} ({ratio:.2f}%)")
    
    print(f"\nTotal labels: {total_labels:,}")
    
    weights = []
    for cls in range(num_classes):
        count = label_counts.get(cls, 1)
        weight = total_labels / (num_classes * count)
        weights.append(weight)
    
    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum() * num_classes
    
    print(f"\nClass weights (balanced):")
    for cls, w in enumerate(weights):
        print(f"  Class {cls}: {w:.4f}")
    
    weights_file = data_path / "class_weights.pt"
    torch.save(weights, weights_file)
    print(f"\nSaved to {weights_file}")
    
    return weights


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()
    
    compute_class_weights(args.data_dir, args.max_samples)