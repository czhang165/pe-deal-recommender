"""
Ray Train distributed training module.

This module provides distributed training using Ray Train for scaling
training across multiple GPUs/nodes.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F

from src.config import Settings, get_settings

# Ray imports are optional - only needed when using distributed training
try:
    import ray
    from ray import train
    from ray.train import ScalingConfig, RunConfig, CheckpointConfig
    from ray.train.torch import TorchTrainer
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False


def check_ray_available() -> None:
    """Raise error if Ray is not installed."""
    if not RAY_AVAILABLE:
        raise ImportError(
            "Ray is not installed. Install with: pip install 'pe-deal-recommender[ray]'"
        )


def init_ray(settings: Optional[Settings] = None) -> None:
    """
    Initialize Ray cluster.
    
    If ray_address is None, starts a local cluster.
    If ray_address is "auto", connects to existing cluster.
    """
    check_ray_available()
    
    if settings is None:
        settings = get_settings()
    
    if ray.is_initialized():
        return
    
    if settings.ray_address:
        ray.init(address=settings.ray_address, namespace=settings.ray_namespace)
    else:
        # Start local cluster
        ray.init(namespace=settings.ray_namespace)
    
    print(f"Ray initialized: {ray.cluster_resources()}")


def train_two_tower_distributed(
    train_dataset_path: str,
    val_dataset_path: str,
    n_investors: int,
    n_deals: int,
    feature_dims: dict,
    settings: Optional[Settings] = None,
) -> Any:
    """
    Train Two-Tower model using Ray distributed training.
    
    Args:
        train_dataset_path: Path to training data
        val_dataset_path: Path to validation data
        n_investors: Number of investors
        n_deals: Number of deals
        feature_dims: Feature dimension dict
        settings: Configuration settings
        
    Returns:
        Ray Train Result object
    """
    check_ray_available()
    
    if settings is None:
        settings = get_settings()
    
    init_ray(settings)
    
    def train_loop_per_worker(config: dict) -> None:
        """Training function executed on each worker."""
        from src.models import TwoTowerModel
        from src.config import TwoTowerConfig, FeatureDims
        
        # Get data shards for this worker
        train_shard = train.get_dataset_shard("train")
        val_shard = train.get_dataset_shard("val")
        
        # Initialize model
        model_config = TwoTowerConfig(**config.get("model_config", {}))
        model = TwoTowerModel(
            n_investors=config["n_investors"],
            n_deals=config["n_deals"],
            feature_dims=FeatureDims(**config["feature_dims"]),
            config=model_config,
            learning_rate=config.get("lr", 1e-3),
        )
        
        # Prepare model for distributed training
        model = train.torch.prepare_model(model)
        
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.get("lr", 1e-3)
        )
        
        # Training loop
        for epoch in range(config.get("epochs", 30)):
            model.train()
            train_loss = 0.0
            train_steps = 0
            
            for batch in train_shard.iter_torch_batches(
                batch_size=config.get("batch_size", 128)
            ):
                investor_data, deal_data, labels = batch
                
                optimizer.zero_grad()
                similarities = model(investor_data, deal_data)
                loss = F.binary_cross_entropy_with_logits(similarities, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_steps += 1
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_steps = 0
            
            with torch.no_grad():
                for batch in val_shard.iter_torch_batches(
                    batch_size=config.get("batch_size", 128)
                ):
                    investor_data, deal_data, labels = batch
                    similarities = model(investor_data, deal_data)
                    loss = F.binary_cross_entropy_with_logits(similarities, labels)
                    val_loss += loss.item()
                    val_steps += 1
            
            avg_train_loss = train_loss / max(train_steps, 1)
            avg_val_loss = val_loss / max(val_steps, 1)
            
            # Report metrics and checkpoint
            with tempfile.TemporaryDirectory() as tmpdir:
                checkpoint_path = Path(tmpdir) / "model.pt"
                torch.save({
                    "model_state_dict": model.module.state_dict() 
                        if hasattr(model, "module") else model.state_dict(),
                    "epoch": epoch,
                }, checkpoint_path)
                
                train.report(
                    metrics={
                        "train_loss": avg_train_loss,
                        "val_loss": avg_val_loss,
                        "epoch": epoch,
                    },
                    checkpoint=train.Checkpoint.from_directory(tmpdir),
                )
    
    # Create Ray datasets
    train_dataset = _create_ray_dataset(train_dataset_path)
    val_dataset = _create_ray_dataset(val_dataset_path)
    
    # Configure trainer
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={
            "n_investors": n_investors,
            "n_deals": n_deals,
            "feature_dims": feature_dims,
            "lr": settings.training.learning_rate,
            "batch_size": settings.training.batch_size,
            "epochs": settings.training.max_epochs,
            "model_config": settings.two_tower.model_dump(),
        },
        scaling_config=ScalingConfig(
            num_workers=settings.ray_training.num_workers,
            use_gpu=settings.ray_training.use_gpu,
            resources_per_worker=settings.ray_training.resources_per_worker,
        ),
        run_config=RunConfig(
            name="two-tower-distributed",
            storage_path=str(settings.models_dir / "ray_results"),
            checkpoint_config=CheckpointConfig(
                num_to_keep=settings.training.save_top_k,
                checkpoint_score_attribute="val_loss",
                checkpoint_score_order="min",
            ),
        ),
        datasets={"train": train_dataset, "val": val_dataset},
    )
    
    result = trainer.fit()
    return result


def train_deep_ranking_distributed(
    train_dataset_path: str,
    val_dataset_path: str,
    n_investors: int,
    n_deals: int,
    feature_dims: dict,
    settings: Optional[Settings] = None,
) -> Any:
    """
    Train Deep Ranking model using Ray distributed training.
    
    Similar to train_two_tower_distributed but for pairwise ranking model.
    """
    check_ray_available()
    
    if settings is None:
        settings = get_settings()
    
    init_ray(settings)
    
    def train_loop_per_worker(config: dict) -> None:
        """Training function for deep ranking."""
        from src.models import PairwiseDeepRankingModel
        from src.config import DeepRankingConfig, FeatureDims
        
        train_shard = train.get_dataset_shard("train")
        val_shard = train.get_dataset_shard("val")
        
        model_config = DeepRankingConfig(**config.get("model_config", {}))
        model = PairwiseDeepRankingModel(
            n_investors=config["n_investors"],
            n_deals=config["n_deals"],
            feature_dims=FeatureDims(**config["feature_dims"]),
            config=model_config,
            learning_rate=config.get("lr", 1e-3),
        )
        
        model = train.torch.prepare_model(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.get("lr", 1e-3))
        
        for epoch in range(config.get("epochs", 30)):
            model.train()
            train_loss = 0.0
            train_steps = 0
            
            for batch in train_shard.iter_torch_batches(
                batch_size=config.get("batch_size", 128)
            ):
                investor_data, better_deal_data, worse_deal_data = batch
                
                optimizer.zero_grad()
                better_scores = model(investor_data, better_deal_data).squeeze(-1)
                worse_scores = model(investor_data, worse_deal_data).squeeze(-1)
                
                loss = F.binary_cross_entropy_with_logits(
                    better_scores - worse_scores,
                    torch.ones_like(better_scores)
                )
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_steps += 1
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_steps = 0
            
            with torch.no_grad():
                for batch in val_shard.iter_torch_batches(
                    batch_size=config.get("batch_size", 128)
                ):
                    investor_data, better_deal_data, worse_deal_data = batch
                    better_scores = model(investor_data, better_deal_data).squeeze(-1)
                    worse_scores = model(investor_data, worse_deal_data).squeeze(-1)
                    
                    loss = F.binary_cross_entropy_with_logits(
                        better_scores - worse_scores,
                        torch.ones_like(better_scores)
                    )
                    val_loss += loss.item()
                    val_steps += 1
            
            avg_train_loss = train_loss / max(train_steps, 1)
            avg_val_loss = val_loss / max(val_steps, 1)
            
            with tempfile.TemporaryDirectory() as tmpdir:
                checkpoint_path = Path(tmpdir) / "model.pt"
                torch.save({
                    "model_state_dict": model.module.state_dict() 
                        if hasattr(model, "module") else model.state_dict(),
                    "epoch": epoch,
                }, checkpoint_path)
                
                train.report(
                    metrics={
                        "train_loss": avg_train_loss,
                        "val_loss": avg_val_loss,
                        "epoch": epoch,
                    },
                    checkpoint=train.Checkpoint.from_directory(tmpdir),
                )
    
    train_dataset = _create_ray_dataset(train_dataset_path, pairwise=True)
    val_dataset = _create_ray_dataset(val_dataset_path, pairwise=True)
    
    trainer = TorchTrainer(
        train_loop_per_worker=train_loop_per_worker,
        train_loop_config={
            "n_investors": n_investors,
            "n_deals": n_deals,
            "feature_dims": feature_dims,
            "lr": settings.training.learning_rate,
            "batch_size": settings.training.batch_size,
            "epochs": settings.training.max_epochs,
            "model_config": settings.deep_ranking.model_dump(),
        },
        scaling_config=ScalingConfig(
            num_workers=settings.ray_training.num_workers,
            use_gpu=settings.ray_training.use_gpu,
            resources_per_worker=settings.ray_training.resources_per_worker,
        ),
        run_config=RunConfig(
            name="deep-ranking-distributed",
            storage_path=str(settings.models_dir / "ray_results"),
            checkpoint_config=CheckpointConfig(
                num_to_keep=settings.training.save_top_k,
                checkpoint_score_attribute="val_loss",
                checkpoint_score_order="min",
            ),
        ),
        datasets={"train": train_dataset, "val": val_dataset},
    )
    
    result = trainer.fit()
    return result


def _create_ray_dataset(path: str, pairwise: bool = False) -> Any:
    """
    Create a Ray Dataset from file path.
    
    This is a placeholder - in production you would implement proper
    data loading from your storage backend.
    """
    check_ray_available()
    
    # For now, return a simple dataset from parquet/csv
    # In production, implement proper data pipeline
    if path.endswith(".parquet"):
        return ray.data.read_parquet(path)
    elif path.endswith(".csv"):
        return ray.data.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {path}")


def shutdown_ray() -> None:
    """Shutdown Ray cluster."""
    if RAY_AVAILABLE and ray.is_initialized():
        ray.shutdown()
