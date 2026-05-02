from typing import Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from lightning.pytorch import LightningModule
from torchmetrics.classification import MulticlassF1Score, MulticlassPrecision, MulticlassRecall


class PunctuationTrainer(LightningModule):
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        warmup_steps: int = 100,
        num_training_steps: int = 10000,
        class_weights: Optional[torch.Tensor] = None,
        num_labels: int = 5,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.num_training_steps = num_training_steps
        self.num_labels = num_labels
        
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        self.val_precision = MulticlassPrecision(num_classes=num_labels, ignore_index=-100, average="none")
        self.val_recall = MulticlassRecall(num_classes=num_labels, ignore_index=-100, average="none")
        self.val_f1 = MulticlassF1Score(num_classes=num_labels, ignore_index=-100, average="none")
        self.val_macro_f1 = MulticlassF1Score(num_classes=num_labels, ignore_index=-100, average="macro")
        
        self.test_precision = MulticlassPrecision(num_classes=num_labels, ignore_index=-100, average="none")
        self.test_recall = MulticlassRecall(num_classes=num_labels, ignore_index=-100, average="none")
        self.test_f1 = MulticlassF1Score(num_classes=num_labels, ignore_index=-100, average="none")
        self.test_macro_f1 = MulticlassF1Score(num_classes=num_labels, ignore_index=-100, average="macro")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict:
        return self.model(input_ids, attention_mask, labels)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        logits = outputs["logits"]
        labels = batch["labels"]
        
        loss = self.criterion(
            logits.view(-1, self.num_labels),
            labels.view(-1),
        )

        preds = torch.argmax(logits, dim=-1)
        mask = labels != -100
        correct = ((preds == labels) & mask).sum().float()
        total = mask.sum().float()
        accuracy = correct / total if total > 0 else torch.tensor(0.0, device=self.device)

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        self.log("train/accuracy", accuracy, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"], on_step=True, logger=True)

        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        logits = outputs["logits"]
        labels = batch["labels"]

        loss = self.criterion(
            logits.view(-1, self.num_labels),
            labels.view(-1),
        )

        # Flatten predictions and labels for metric calculation
        preds = torch.argmax(logits, dim=-1).view(-1)
        labels_flat = labels.view(-1)

        self.val_precision.update(preds, labels_flat)
        self.val_recall.update(preds, labels_flat)
        self.val_f1.update(preds, labels_flat)
        self.val_macro_f1.update(preds, labels_flat)

        self.log("val/loss", loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)

        return loss

    def on_validation_epoch_end(self):
        precision = self.val_precision.compute()
        recall = self.val_recall.compute()
        f1 = self.val_f1.compute()
        macro_f1 = self.val_macro_f1.compute()
        
        for cls in range(self.num_labels):
            self.log(f"val/class{cls}_precision", precision[cls], logger=True)
            self.log(f"val/class{cls}_recall", recall[cls], logger=True)
            self.log(f"val/class{cls}_f1", f1[cls], logger=True)
        
        self.log("val/macro_f1", macro_f1, prog_bar=True, logger=True)
        
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()
        self.val_macro_f1.reset()

    def test_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        outputs = self(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )

        logits = outputs["logits"]
        labels = batch["labels"]

        loss = self.criterion(
            logits.view(-1, self.num_labels),
            labels.view(-1),
        )

        # Flatten predictions and labels for metric calculation
        preds = torch.argmax(logits, dim=-1).view(-1)
        labels_flat = labels.view(-1)

        self.test_precision.update(preds, labels_flat)
        self.test_recall.update(preds, labels_flat)
        self.test_f1.update(preds, labels_flat)
        self.test_macro_f1.update(preds, labels_flat)

        self.log("test/loss", loss, prog_bar=True, logger=True)

        return loss

    def on_test_epoch_end(self):
        precision = self.test_precision.compute()
        recall = self.test_recall.compute()
        f1 = self.test_f1.compute()
        macro_f1 = self.test_macro_f1.compute()
        
        for cls in range(self.num_labels):
            self.log(f"test/class{cls}_precision", precision[cls], logger=True)
            self.log(f"test/class{cls}_recall", recall[cls], logger=True)
            self.log(f"test/class{cls}_f1", f1[cls], logger=True)
        
        self.log("test/macro_f1", macro_f1, prog_bar=True, logger=True)
        
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_f1.reset()
        self.test_macro_f1.reset()

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        pct_start = min(0.1, self.warmup_steps / max(self.num_training_steps, 1))

        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=self.num_training_steps,
            pct_start=pct_start,
            anneal_strategy="cos",
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }