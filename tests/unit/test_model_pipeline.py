"""
DeepGuard — tests/unit/test_model_pipeline.py

Unit tests for classification heads, ViT wrapper, focal loss, warmup scheduler,
metrics evaluator, and trainer.
"""

from pathlib import Path
import pytest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from models.heads.linear import LinearClassifierHead
from models.heads.mlp import MLPClassifierHead
from models.heads.attention_pool import AttentionPoolingHead
from models.architectures.vit import ViTClassifier
from training.losses.focal import FocalLoss
from training.schedulers.cosine_warmup import CosineAnnealingWithWarmupLR
from training.evaluators.metrics import MetricsEvaluator
from training.trainer import Trainer
from training.callbacks.base import Callback
from training.callbacks.early_stopping import EarlyStoppingCallback
from training.callbacks.checkpointing import ModelCheckpointCallback
from training.callbacks.tensorboard_logger import TensorBoardCallback
from training.callbacks.mlflow_logger import MLflowCallback


def test_classification_heads() -> None:
    # Batch size = 4, embedding features = 128, num_classes = 2, num_tokens = 10
    x_pooled = torch.randn(4, 128)
    x_seq = torch.randn(4, 10, 128)

    # 1. Linear Head
    linear_head = LinearClassifierHead(in_features=128, num_classes=2, dropout=0.1)
    out_lp = linear_head(x_pooled)
    out_ls = linear_head(x_seq)
    assert out_lp.shape == (4, 2)
    assert out_ls.shape == (4, 2)

    # 2. MLP Head
    mlp_head = MLPClassifierHead(in_features=128, num_classes=2, hidden_dim=64, dropout=0.2)
    out_mp = mlp_head(x_pooled)
    out_ms = mlp_head(x_seq)
    assert out_mp.shape == (4, 2)
    assert out_ms.shape == (4, 2)

    # 3. Attention Pooling Head
    att_head = AttentionPoolingHead(in_features=128, num_classes=2, num_heads=4, dropout=0.1)
    out_ap = att_head(x_pooled)
    out_as = att_head(x_seq)
    assert out_ap.shape == (4, 2)
    assert out_as.shape == (4, 2)


def test_vit_classifier_strategies() -> None:
    # Instantiate tiny ViT backbone to run tests quickly without downloading large files
    model = ViTClassifier(
        model_name="vit_tiny_patch16_224",
        pretrained=False,
        num_classes=2,
        head_type="linear",
        fine_tuning_strategy="head_only",
    )

    # Verify frozen parameters in head_only strategy
    for name, param in model.backbone.named_parameters():
        assert not param.requires_grad

    for name, param in model.head.named_parameters():
        assert param.requires_grad

    # Check forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    logits = model(dummy_input)
    assert logits.shape == (2, 2)


def test_focal_loss() -> None:
    loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
    logits = torch.tensor([[1.5, -0.5], [-1.0, 2.0]], dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.int64)

    loss = loss_fn(logits, targets)
    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0.0


def test_cosine_warmup_scheduler() -> None:
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    scheduler = CosineAnnealingWithWarmupLR(
        optimizer=optimizer,
        warmup_steps=10,
        total_steps=100,
        min_lr=1e-6,
    )

    # Initial learning rate should be close to min_lr
    assert abs(scheduler.get_last_lr()[0] - 1e-6) < 1e-8

    # Warmup ramp up
    for _ in range(10):
        scheduler.step()
    assert abs(scheduler.get_last_lr()[0] - 1e-3) < 1e-5

    # Decay ramp down
    for _ in range(50):
        scheduler.step()
    assert scheduler.get_last_lr()[0] < 1e-3
    assert scheduler.get_last_lr()[0] > 1e-6


def test_metrics_evaluator() -> None:
    y_true = [0, 1, 0, 1, 1]
    y_pred_probs = [0.1, 0.9, 0.4, 0.3, 0.8]  # binary prediction threshold at 0.5: [0, 1, 0, 0, 1]

    metrics = MetricsEvaluator.evaluate(y_true, y_pred_probs)

    assert metrics["accuracy"] == 0.8  # 4 out of 5 correct
    assert metrics["confusion_matrix"]["tp"] == 2
    assert metrics["confusion_matrix"]["fp"] == 0
    assert metrics["confusion_matrix"]["tn"] == 2
    assert metrics["confusion_matrix"]["fn"] == 1
    assert "roc_curve" in metrics
    assert "auc_roc" in metrics


def test_trainer_fit(tmp_path: Path) -> None:
    # A simple linear model for rapid fit testing
    class DummyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(10, 2)
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)

    model = DummyModel()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Build dummy dataset
    x = torch.randn(20, 10)
    y = torch.randint(0, 2, (20,))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=4)

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        callbacks=None,
        device="cpu",
        use_amp=False,
        output_dir=tmp_path,
    )

    metrics = trainer.fit(train_loader=loader, val_loader=loader, epochs=2)
    assert "train/loss" in metrics
    assert "val/loss" in metrics
    assert "val/accuracy" in metrics


def test_early_stopping_callback() -> None:
    class DummyTrainer:
        def __init__(self) -> None:
            self.stop_training = False

    trainer = DummyTrainer()
    cb = EarlyStoppingCallback(monitor="val_auc", patience=2, mode="max")

    # Epoch 1: improve
    cb.on_epoch_end(trainer, 0, {"val_auc": 0.8})
    assert not trainer.stop_training
    assert cb.best == 0.8
    assert cb.wait == 0

    # Epoch 2: no improve
    cb.on_epoch_end(trainer, 1, {"val_auc": 0.79})
    assert not trainer.stop_training
    assert cb.wait == 1

    # Epoch 3: no improve, triggers stopping
    cb.on_epoch_end(trainer, 2, {"val_auc": 0.79})
    assert trainer.stop_training
    assert cb.wait == 2


def test_checkpointing_callback(tmp_path: Path) -> None:
    class DummyTrainer:
        def __init__(self) -> None:
            self.model = nn.Linear(5, 2)
            self.optimizer = None
            self.scheduler = None
            self.output_dir = tmp_path

    trainer = DummyTrainer()
    cb = ModelCheckpointCallback(
        output_dir=tmp_path,
        monitor="val_auc",
        mode="max",
        save_top_k=2,
        filename_template="epoch={epoch:02d}-val={val_auc:.2f}",
    )

    # Epoch 1
    cb.on_epoch_end(trainer, 0, {"val_auc": 0.85})
    assert (tmp_path / "best_model.pt").exists()
    assert (tmp_path / "epoch=01-val=0.85.pt").exists()

    # Epoch 2: better
    cb.on_epoch_end(trainer, 1, {"val_auc": 0.90})
    assert (tmp_path / "epoch=02-val=0.90.pt").exists()

    # Epoch 3: worse
    cb.on_epoch_end(trainer, 2, {"val_auc": 0.75})
    # Since save_top_k=2, epoch 3 (0.75) is immediately deleted as it's worse than epoch 2 (0.90) and epoch 1 (0.85).
    assert not (tmp_path / "epoch=03-val=0.75.pt").exists()

    # Epoch 4: even worse (should delete epoch 4 as well, keeping epoch=02 and epoch=01)
    cb.on_epoch_end(trainer, 3, {"val_auc": 0.70})
    assert not (tmp_path / "epoch=04-val=0.70.pt").exists()
    assert (tmp_path / "epoch=02-val=0.90.pt").exists()
    assert (tmp_path / "epoch=01-val=0.85.pt").exists()


def test_tensorboard_callback(tmp_path: Path) -> None:
    cb = TensorBoardCallback(log_dir=str(tmp_path))
    class DummyTrainer:
        pass
    trainer = DummyTrainer()
    cb.on_train_begin(trainer)
    cb.on_epoch_end(trainer, 0, {"train/loss": 0.5, "val/loss": 0.4})
    cb.on_train_end(trainer)
    assert len(list(tmp_path.glob("events.out.tfevents.*"))) > 0


def test_mlflow_callback() -> None:
    cb = MLflowCallback(experiment_name="test-exp", run_name="test-run", params={"lr": 0.01})
    class DummyTrainer:
        def __init__(self) -> None:
            self.output_dir = "."
    trainer = DummyTrainer()
    cb.on_train_begin(trainer)
    cb.on_epoch_end(trainer, 0, {"val_loss": 0.1})
    cb.on_train_end(trainer)

