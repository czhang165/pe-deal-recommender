"""
Local training module using PyTorch Lightning.

This module provides the standard (non-distributed) training pipeline
for both Two-Tower and Deep Ranking models.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    RichProgressBar,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from src.config import Settings, get_settings
from src.data import DataBundle, load_data, create_dataloaders, FeatureEncoder
from src.models import TwoTowerModel, PairwiseDeepRankingModel


class TrainingResult:
    """Container for training results."""
    
    def __init__(
        self,
        model_type: str,
        best_checkpoint_path: str,
        final_metrics: dict,
        training_config: dict,
        data_config: dict,
    ):
        self.model_type = model_type
        self.best_checkpoint_path = best_checkpoint_path
        self.final_metrics = final_metrics
        self.training_config = training_config
        self.data_config = data_config
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> dict:
        return {
            "model_type": self.model_type,
            "best_checkpoint_path": self.best_checkpoint_path,
            "final_metrics": self.final_metrics,
            "training_config": self.training_config,
            "data_config": self.data_config,
            "timestamp": self.timestamp,
        }
    
    def save(self, path: str | Path) -> None:
        """Save training result to JSON."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


def train_two_tower(
    data_bundle: Optional[DataBundle] = None,
    settings: Optional[Settings] = None,
) -> TrainingResult:
    """
    Train the Two-Tower model.
    
    Args:
        data_bundle: Pre-loaded data (loads if not provided)
        settings: Configuration settings
        
    Returns:
        TrainingResult with checkpoint path and metrics
    """
    if settings is None:
        settings = get_settings()
    
    if data_bundle is None:
        print("Loading data...")
        data_bundle = load_data(settings)
    
    print("Creating dataloaders...")
    train_loader, val_loader, _ = create_dataloaders(
        data_bundle, settings, model_type="two_tower"
    )
    
    print("Initializing Two-Tower model...")
    model = TwoTowerModel(
        n_investors=data_bundle.n_investors,
        n_deals=data_bundle.n_deals,
        feature_dims=data_bundle.encoder.feature_dims,
        config=settings.two_tower,
    )
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=settings.models_dir / "checkpoints",
        filename="two-tower-{epoch:02d}-{val_loss:.3f}",
        save_top_k=settings.training.save_top_k,
        monitor=settings.training.checkpoint_monitor,
        mode=settings.training.checkpoint_mode,
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=settings.training.early_stopping_patience,
        min_delta=settings.training.early_stopping_min_delta,
        mode="min",
    )
    
    callbacks = [
        checkpoint_callback,
        early_stop_callback,
        LearningRateMonitor(logging_interval="step"),
    ]
    
    # Try to use rich progress bar if available
    try:
        callbacks.append(RichProgressBar())
    except Exception:
        pass
    
    # Setup loggers
    loggers = [
        TensorBoardLogger(settings.logs_dir, name="two_tower"),
        CSVLogger(settings.logs_dir, name="two_tower_csv"),
    ]
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=settings.training.max_epochs,
        accelerator=settings.training.accelerator,
        devices=settings.training.devices,
        precision=settings.training.precision,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=settings.training.log_every_n_steps,
        enable_model_summary=True,
    )
    
    # Train
    print("Starting Two-Tower training...")
    trainer.fit(model, train_loader, val_loader)
    
    # Get best metrics
    final_metrics = {
        "best_val_loss": float(checkpoint_callback.best_model_score or 0),
        "epochs_trained": trainer.current_epoch,
    }
    
    return TrainingResult(
        model_type="two_tower",
        best_checkpoint_path=checkpoint_callback.best_model_path,
        final_metrics=final_metrics,
        training_config=settings.training.model_dump(),
        data_config={
            "n_investors": data_bundle.n_investors,
            "n_deals": data_bundle.n_deals,
            "n_train": len(data_bundle.train_interactions),
            "n_val": len(data_bundle.val_interactions),
            "n_test": len(data_bundle.test_interactions),
            "feature_dims": data_bundle.encoder.feature_dims.model_dump(),
        },
    )


def train_deep_ranking(
    data_bundle: Optional[DataBundle] = None,
    settings: Optional[Settings] = None,
) -> TrainingResult:
    """
    Train the Deep Ranking model.
    
    Args:
        data_bundle: Pre-loaded data (loads if not provided)
        settings: Configuration settings
        
    Returns:
        TrainingResult with checkpoint path and metrics
    """
    if settings is None:
        settings = get_settings()
    
    if data_bundle is None:
        print("Loading data...")
        data_bundle = load_data(settings)
    
    print("Creating pairwise dataloaders...")
    train_loader, val_loader, _ = create_dataloaders(
        data_bundle, settings, model_type="deep_ranking"
    )
    
    print("Initializing Deep Ranking model...")
    model = PairwiseDeepRankingModel(
        n_investors=data_bundle.n_investors,
        n_deals=data_bundle.n_deals,
        feature_dims=data_bundle.encoder.feature_dims,
        config=settings.deep_ranking,
    )
    
    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=settings.models_dir / "checkpoints",
        filename="deep-ranking-{epoch:02d}-{val_loss:.3f}",
        save_top_k=settings.training.save_top_k,
        monitor=settings.training.checkpoint_monitor,
        mode=settings.training.checkpoint_mode,
    )
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=settings.training.early_stopping_patience,
        min_delta=settings.training.early_stopping_min_delta,
        mode="min",
    )
    
    callbacks = [
        checkpoint_callback,
        early_stop_callback,
        LearningRateMonitor(logging_interval="step"),
    ]
    
    try:
        callbacks.append(RichProgressBar())
    except Exception:
        pass
    
    # Setup loggers
    loggers = [
        TensorBoardLogger(settings.logs_dir, name="deep_ranking"),
        CSVLogger(settings.logs_dir, name="deep_ranking_csv"),
    ]
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=settings.training.max_epochs,
        accelerator=settings.training.accelerator,
        devices=settings.training.devices,
        precision=settings.training.precision,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=settings.training.log_every_n_steps,
        enable_model_summary=True,
    )
    
    # Train
    print("Starting Deep Ranking training...")
    trainer.fit(model, train_loader, val_loader)
    
    # Get best metrics
    final_metrics = {
        "best_val_loss": float(checkpoint_callback.best_model_score or 0),
        "epochs_trained": trainer.current_epoch,
    }
    
    return TrainingResult(
        model_type="deep_ranking",
        best_checkpoint_path=checkpoint_callback.best_model_path,
        final_metrics=final_metrics,
        training_config=settings.training.model_dump(),
        data_config={
            "n_investors": data_bundle.n_investors,
            "n_deals": data_bundle.n_deals,
            "n_train": len(data_bundle.train_interactions),
            "n_val": len(data_bundle.val_interactions),
            "n_test": len(data_bundle.test_interactions),
            "feature_dims": data_bundle.encoder.feature_dims.model_dump(),
        },
    )


def train_all(
    settings: Optional[Settings] = None,
    save_summary: bool = True,
) -> dict[str, TrainingResult]:
    """
    Train both Two-Tower and Deep Ranking models.
    
    Args:
        settings: Configuration settings
        save_summary: Whether to save training summary to disk
        
    Returns:
        Dictionary with results for each model
    """
    if settings is None:
        settings = get_settings()
    
    # Load data once for both models
    print("=" * 60)
    print("Loading data...")
    print("=" * 60)
    data_bundle = load_data(settings)
    
    # Save feature encoder
    encoder_path = settings.models_dir / "feature_encoder.pkl"
    data_bundle.encoder.save(encoder_path)
    print(f"Saved feature encoder to {encoder_path}")
    
    results = {}
    
    # Train Two-Tower
    print("\n" + "=" * 60)
    print("Training Two-Tower Model")
    print("=" * 60)
    results["two_tower"] = train_two_tower(data_bundle, settings)
    
    # Train Deep Ranking
    print("\n" + "=" * 60)
    print("Training Deep Ranking Model")
    print("=" * 60)
    results["deep_ranking"] = train_deep_ranking(data_bundle, settings)
    
    # Save training summary
    if save_summary:
        summary = {
            "two_tower": results["two_tower"].to_dict(),
            "deep_ranking": results["deep_ranking"].to_dict(),
            "feature_encoder_path": str(encoder_path),
        }
        summary_path = settings.models_dir / "training_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nTraining summary saved to {summary_path}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Two-Tower best checkpoint: {results['two_tower'].best_checkpoint_path}")
    print(f"Deep Ranking best checkpoint: {results['deep_ranking'].best_checkpoint_path}")
    
    return results
