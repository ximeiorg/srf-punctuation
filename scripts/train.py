import argparse
from pathlib import Path

import torch
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import Trainer

from srf_punctuation.config import Config
from srf_punctuation.data import DataProcessor, PunctuationDataModule
from srf_punctuation.models import PunctuationPredictor
from srf_punctuation.trainers import PunctuationTrainer


# Load default config for argument defaults
_DEFAULT_CONFIG = Config()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train punctuation prediction model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--process-data",
        action="store_true",
        help="Process raw data before training",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=_DEFAULT_CONFIG.training.batch_size,
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=_DEFAULT_CONFIG.training.learning_rate,
        help="Learning rate",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=_DEFAULT_CONFIG.training.max_epochs,
        help="Maximum number of epochs",
    )
    parser.add_argument(
        "--embed-dim",
        type=int,
        default=_DEFAULT_CONFIG.model.embed_dim,
        help="Embedding dimension",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=_DEFAULT_CONFIG.model.hidden_dim,
        help="Hidden dimension",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=_DEFAULT_CONFIG.model.num_layers,
        help="Number of transformer layers",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=_DEFAULT_CONFIG.data.max_samples,
        help="Maximum number of samples to use (None for unlimited)",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="Number of GPUs to use",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config = Config()
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.learning_rate
    config.training.max_epochs = args.max_epochs
    config.model.embed_dim = args.embed_dim
    config.model.hidden_dim = args.hidden_dim
    config.model.num_layers = args.num_layers
    config.data.max_samples = args.max_samples

    data_dir = Path(config.data.processed_data_dir)
    data_exists = (
        (data_dir / config.data.train_file).exists()
        and (data_dir / config.data.val_file).exists()
        and (data_dir / config.data.test_file).exists()
        and (data_dir / config.data.vocab_file).exists()
    )

    if args.process_data or not data_exists:
        if data_exists and not args.process_data:
            print("Data files missing, processing raw data...")
        else:
            print("Processing raw data...")
        processor = DataProcessor(config)
        processor.process_and_save()

    weights_path = data_dir / "class_weights.pt"
    if not weights_path.exists():
        print("Computing class weights...")
        import subprocess
        subprocess.run([
            "uv", "run", "python", "scripts/compute_class_weights.py",
            "--data-dir", config.data.processed_data_dir,
        ], check=True)

    data_module = PunctuationDataModule(
        data_dir=config.data.processed_data_dir,
        max_seq_len=config.model.max_seq_len,
        batch_size=config.training.batch_size,
        num_workers=0,
    )
    data_module.setup()

    model = PunctuationPredictor(
        vocab_size=config.model.vocab_size,
        embed_dim=config.model.embed_dim,
        hidden_dim=config.model.hidden_dim,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        num_labels=config.model.num_labels,
        dropout=config.model.dropout,
        max_seq_len=config.model.max_seq_len,
    )

    print(f"Model parameters: {model.count_parameters():,}")

    import math
    steps_per_epoch = math.ceil(len(data_module.train_dataset) / config.training.batch_size)
    num_training_steps = steps_per_epoch * config.training.max_epochs

    class_weights = None
    weights_path = Path(config.data.processed_data_dir) / "class_weights.pt"
    if weights_path.exists():
        class_weights = torch.load(weights_path, weights_only=True)
        print(f"Loaded class weights from {weights_path}")

    trainer_module = PunctuationTrainer(
        model=model,
        learning_rate=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        warmup_steps=config.training.warmup_steps,
        num_training_steps=num_training_steps,
        class_weights=class_weights,
        num_labels=config.model.num_labels,
    )

    checkpoint_callback = ModelCheckpoint(
        filename="punctuation-{epoch:02d}-{val/loss:.4f}",
        save_top_k=3,
        monitor="val/loss",
        mode="min",
        save_last=True,
    )

    early_stop_callback = EarlyStopping(
        monitor="val/loss",
        patience=config.training.early_stopping_patience,
        mode="min",
    )

    tensorboard_logger = TensorBoardLogger(
        save_dir=config.training.log_dir,
        name="punctuation",
    )

    trainer = Trainer(
        max_epochs=config.training.max_epochs,
        accelerator="auto",
        devices=args.gpus if torch.cuda.is_available() else 1,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=tensorboard_logger,
        gradient_clip_val=config.training.gradient_clip_val,
        log_every_n_steps=10,
    )

    trainer.fit(
        trainer_module,
        train_dataloaders=data_module.train_dataloader(),
        val_dataloaders=data_module.val_dataloader(),
    )

    trainer.test(dataloaders=data_module.test_dataloader())

    best_model_path = checkpoint_callback.best_model_path
    print(f"Best model saved at: {best_model_path}")
    print(f"TensorBoard logs saved at: {config.training.log_dir}/punctuation")
    print(f"View TensorBoard: tensorboard --logdir={config.training.log_dir}")

    save_path = Path(best_model_path).parent / "model.onnx"
    model.eval()
    model.export_onnx(str(save_path))


if __name__ == "__main__":
    main()