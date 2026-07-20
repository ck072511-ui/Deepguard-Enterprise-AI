"""
DeepGuard — scripts/train.py

CLI orchestrator script to train Vision Transformer deepfake detection models.
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml

# Adjust Python path to import from project root
sys.path.append(str(Path(__file__).resolve().parents[1]))

from datasets.factory import DatasetFactory
from models.factory import ModelFactory
from training.callbacks.checkpointing import ModelCheckpointCallback
from training.callbacks.early_stopping import EarlyStoppingCallback
from training.callbacks.mlflow_logger import MLflowCallback
from training.callbacks.tensorboard_logger import TensorBoardCallback
from training.losses.focal import FocalLoss
from training.schedulers.cosine_warmup import CosineAnnealingWithWarmupLR
from training.trainer import Trainer

logger = logging.getLogger("deepguard.train")


def setup_logging() -> None:
    """Initialize simple training script logs."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(
        description="DeepGuard Vision Transformer Training Script"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training_config.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument(
        "--config-model",
        type=str,
        default="configs/model_config.yaml",
        help="Path to model config YAML",
    )
    parser.add_argument(
        "--config-dataset",
        type=str,
        default="configs/dataset_config.yaml",
        help="Path to dataset config YAML",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training epochs count",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override training batch size",
    )
    parser.add_argument(
        "--limit-batches",
        type=int,
        default=None,
        help="Limit number of train/val batches processed (for debugging)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate",
    )

    args = parser.parse_args()

    # Load configs
    with open(args.config) as f:
        train_config = yaml.safe_load(f)
    with open(args.config_model) as f:
        model_config = yaml.safe_load(f)
    with open(args.config_dataset) as f:
        dataset_config = yaml.safe_load(f)

    # Resolve overrides
    training_cfg = train_config.get("training", {})
    if args.epochs is not None:
        training_cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        training_cfg["batch_size"] = args.batch_size

    opt_cfg = train_config.get("optimizer", {})
    if args.lr is not None:
        opt_cfg["lr"] = args.lr

    # Set random seed
    seed = training_cfg.get("seed", 42)
    deterministic = training_cfg.get("deterministic", True)
    set_seed(seed, deterministic)

    # 1. Setup Dataloaders via DatasetFactory
    # Merge custom overrides or configuration into dataset factory config
    # The dataset factory expects 'datasets', 'preprocessing', 'augmentation' keys
    logger.info("Initializing datasets and data loaders...")
    factory = DatasetFactory(dataset_config)

    dataset_name = train_config.get("dataset", {}).get("name", "ff++")
    batch_size = training_cfg.get("batch_size", 32)
    eval_batch_size = training_cfg.get("eval_batch_size", 64)
    num_workers = training_cfg.get("num_workers", 4)
    pin_memory = training_cfg.get("pin_memory", True)
    persistent_workers = training_cfg.get("persistent_workers", True)

    train_loader = factory.create_dataloader(
        dataset_name=dataset_name,
        split="train",
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    val_loader = factory.create_dataloader(
        dataset_name=dataset_name,
        split="val",
        batch_size=eval_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    # 2. Build model via ModelFactory
    logger.info("Building ViT model classifier...")
    model = ModelFactory.create_model(model_config)

    # 3. Loss Function
    loss_cfg = train_config.get("loss", {})
    loss_type = loss_cfg.get("type", "cross_entropy")

    # Resolve class balancing weights if present
    weight_list = loss_cfg.get("class_weights")
    weight = torch.tensor(weight_list, dtype=torch.float32) if weight_list else None

    if loss_type == "focal":
        focal_cfg = loss_cfg.get("focal", {})
        loss_fn: nn.Module = FocalLoss(
            alpha=focal_cfg.get("alpha", 0.25),
            gamma=focal_cfg.get("gamma", 2.0),
            weight=weight,
        )
    elif loss_type == "label_smoothing":
        loss_fn = nn.CrossEntropyLoss(
            label_smoothing=loss_cfg.get("label_smoothing", 0.1),
            weight=weight,
        )
    else:
        loss_fn = nn.CrossEntropyLoss(weight=weight)

    # 4. Optimizer
    opt_type = opt_cfg.get("type", "adamw").lower()
    lr = opt_cfg.get("lr", 1e-4)
    weight_decay = opt_cfg.get("weight_decay", 1e-4)

    if opt_type == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=opt_cfg.get("momentum", 0.9),
            weight_decay=weight_decay,
        )
    else:  # adamw default
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=tuple(opt_cfg.get("betas", [0.9, 0.999])),
            eps=opt_cfg.get("eps", 1e-8),
            amsgrad=opt_cfg.get("amsgrad", False),
        )

    # 5. Learning Rate Scheduler
    sched_cfg = train_config.get("scheduler", {})
    sched_type = sched_cfg.get("type", "cosine_with_warmup").lower()
    scheduler = None

    if sched_type == "cosine_with_warmup":
        total_epochs = training_cfg.get("epochs", 50)
        total_steps = total_epochs * len(train_loader)
        warmup_steps = sched_cfg.get("warmup_steps", 500)
        min_lr = sched_cfg.get("min_lr", 1e-6)

        scheduler = CosineAnnealingWithWarmupLR(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=min_lr,
        )

    # 6. Callbacks Setup
    callbacks = []

    # Checkpoint saving callback
    checkpoint_cfg = train_config.get("checkpointing", {})
    callbacks.append(
        ModelCheckpointCallback(
            output_dir=training_cfg.get("output_dir", "./weights"),
            monitor=checkpoint_cfg.get("monitor", "val/auc_roc"),
            mode=checkpoint_cfg.get("mode", "max"),
            save_top_k=checkpoint_cfg.get("save_top_k", 3),
            filename_template=checkpoint_cfg.get("filename_template", "epoch={epoch:03d}-auc={val_auc_roc:.4f}"),
        )
    )

    # Early stopping callback
    es_cfg = train_config.get("early_stopping", {})
    if es_cfg.get("enabled", True):
        callbacks.append(
            EarlyStoppingCallback(
                monitor=es_cfg.get("monitor", "val/auc_roc"),
                patience=es_cfg.get("patience", 10),
                min_delta=es_cfg.get("min_delta", 1e-4),
                mode=es_cfg.get("mode", "max"),
            )
        )

    # TensorBoard logging callback
    run_name = training_cfg.get("run_name", "vit_base_run")
    tb_log_dir = Path("logs/tensorboard") / run_name
    callbacks.append(TensorBoardCallback(log_dir=str(tb_log_dir)))

    # MLflow logging callback
    mlflow_cfg = train_config.get("mlflow", {})
    if mlflow_cfg.get("enabled", True):
        # Package and merge parameters for experiment tracking
        flat_params = {
            "model": model_config.get("model", {}),
            "optimizer": opt_cfg,
            "loss": loss_cfg,
            "scheduler": sched_cfg,
            "dataset": dataset_name,
            "batch_size": batch_size,
        }
        callbacks.append(
            MLflowCallback(
                experiment_name=mlflow_cfg.get("experiment_name", "deepguard-experiments"),
                run_name=run_name,
                params=flat_params,
            )
        )

    # 7. Trainer Instantiation & Training Loop
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        callbacks=callbacks,
        device="auto",
        use_amp=training_cfg.get("use_amp", True),
        gradient_clip_norm=training_cfg.get("gradient_clip_norm", 1.0),
        output_dir=training_cfg.get("output_dir", "./weights"),
    )

    logger.info("Starting training model fitting...")
    trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=training_cfg.get("epochs", 50),
        limit_batches=args.limit_batches,
    )


if __name__ == "__main__":
    main()
