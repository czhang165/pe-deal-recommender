"""
Configuration management using Pydantic Settings.

Supports loading from environment variables, .env files, and YAML config files.
All settings have sensible defaults for local development.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Deployment environment."""
    LOCAL = "local"
    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


class StorageBackend(str, Enum):
    """Storage backend for models and data."""
    LOCAL = "local"
    S3 = "s3"


class ComputeBackend(str, Enum):
    """Compute backend for training and inference."""
    LOCAL = "local"
    RAY = "ray"


# =============================================================================
# Model Architecture Configs
# =============================================================================

class TwoTowerConfig(BaseModel):
    """Configuration for the Two-Tower retrieval model."""
    
    embedding_dim: int = Field(default=64, ge=16, le=512)
    tower_hidden_dims: list[int] = Field(default=[128, 64])
    temperature: float = Field(default=0.1, gt=0.0, le=1.0)
    dropout_rate: float = Field(default=0.2, ge=0.0, le=0.5)
    l2_normalize: bool = Field(default=True)
    
    @field_validator('tower_hidden_dims')
    @classmethod
    def validate_hidden_dims(cls, v: list[int]) -> list[int]:
        if len(v) < 1:
            raise ValueError("Must have at least one hidden layer")
        if any(d < 1 for d in v):
            raise ValueError("All hidden dimensions must be positive")
        return v


class DeepRankingConfig(BaseModel):
    """Configuration for the Deep Ranking model."""
    
    hidden_dims: list[int] = Field(default=[256, 128, 64])
    dropout_rate: float = Field(default=0.3, ge=0.0, le=0.5)
    use_batch_norm: bool = Field(default=True)
    
    @field_validator('hidden_dims')
    @classmethod
    def validate_hidden_dims(cls, v: list[int]) -> list[int]:
        if len(v) < 1:
            raise ValueError("Must have at least one hidden layer")
        return v


# =============================================================================
# Data Configs
# =============================================================================

class DataConfig(BaseModel):
    """Configuration for data loading and preprocessing."""
    
    # Dataset sizes
    n_investors: int = Field(default=125, ge=1)
    n_deals: int = Field(default=1000, ge=1)
    
    # Negative sampling
    negative_samples: int = Field(default=4, ge=1, le=20)
    pairwise_pairs_per_positive: int = Field(default=5, ge=1, le=20)
    
    # Data splitting
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    val_size: float = Field(default=0.1, gt=0.0, lt=1.0)
    
    # Paths (relative to data_dir)
    interactions_file: str = "enhanced_interactions.csv"
    investor_features_file: str = "investor_features.csv"
    deal_features_file: str = "deal_features.csv"


class FeatureDims(BaseModel):
    """Feature dimensions derived from data."""
    
    n_investor_types: int = Field(default=3)
    n_regions: int = Field(default=4)
    n_risk_profiles: int = Field(default=3)
    n_sectors: int = Field(default=5)
    n_stages: int = Field(default=5)
    
    def to_dict(self) -> dict[str, int]:
        return self.model_dump()


# =============================================================================
# Training Configs
# =============================================================================

class TrainingConfig(BaseModel):
    """Configuration for model training."""
    
    # Basic training params
    batch_size: int = Field(default=128, ge=1)
    learning_rate: float = Field(default=1e-3, gt=0.0)
    weight_decay: float = Field(default=0.0, ge=0.0)
    max_epochs: int = Field(default=30, ge=1)
    
    # Early stopping
    early_stopping_patience: int = Field(default=5, ge=1)
    early_stopping_min_delta: float = Field(default=0.001)
    
    # Checkpointing
    save_top_k: int = Field(default=3, ge=1)
    checkpoint_monitor: str = "val_loss"
    checkpoint_mode: str = "min"
    
    # Hardware
    num_workers: int = Field(default=4, ge=0)
    accelerator: str = Field(default="auto")  # auto, cpu, gpu, tpu
    devices: int | str = Field(default="auto")
    precision: str = Field(default="32")  # 16, 32, bf16
    
    # Logging
    log_every_n_steps: int = Field(default=50, ge=1)


class RayTrainingConfig(BaseModel):
    """Configuration for Ray distributed training."""
    
    num_workers: int = Field(default=2, ge=1)
    use_gpu: bool = Field(default=True)
    resources_per_worker: dict[str, float] = Field(
        default={"CPU": 4, "GPU": 1}
    )
    
    # Ray Tune hyperparameter search
    num_samples: int = Field(default=10, ge=1)
    max_concurrent_trials: int = Field(default=2, ge=1)
    
    # ASHA scheduler params
    asha_max_t: int = Field(default=30)
    asha_grace_period: int = Field(default=5)
    asha_reduction_factor: int = Field(default=2)


# =============================================================================
# Inference Configs
# =============================================================================

class InferenceConfig(BaseModel):
    """Configuration for model inference."""
    
    # Retrieval settings
    retrieval_k: int = Field(default=100, ge=10)
    final_k: int = Field(default=10, ge=1)
    
    # Batch inference
    batch_size: int = Field(default=256, ge=1)
    
    # FAISS settings
    faiss_index_type: str = Field(default="FlatIP")  # FlatIP, IVFFlat, HNSW
    faiss_nprobe: int = Field(default=10, ge=1)  # for IVF indices


class ServingConfig(BaseModel):
    """Configuration for Ray Serve deployment."""
    
    # Replicas
    retriever_replicas: int = Field(default=2, ge=1)
    ranker_replicas: int = Field(default=2, ge=1)
    
    # Resources per replica
    retriever_num_cpus: float = Field(default=2.0)
    retriever_num_gpus: float = Field(default=0.5)
    ranker_num_cpus: float = Field(default=2.0)
    ranker_num_gpus: float = Field(default=0.5)
    
    # Server settings
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1, le=65535)
    
    # Health check
    health_check_period_s: float = Field(default=10.0)
    health_check_timeout_s: float = Field(default=30.0)


# =============================================================================
# Main Settings Class
# =============================================================================

class Settings(BaseSettings):
    """
    Main application settings.
    
    Configuration is loaded in the following priority (highest to lowest):
    1. Environment variables (prefixed with RECSYS_)
    2. .env file
    3. config.yaml file
    4. Default values
    
    Example:
        # From environment
        export RECSYS_ENVIRONMENT=prod
        export RECSYS_DATA_DIR=/data
        
        # From code
        settings = Settings()
        settings = Settings(_env_file=".env.prod")
    """
    
    model_config = SettingsConfigDict(
        env_prefix="RECSYS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
    
    # ==========================================================================
    # Core Settings
    # ==========================================================================
    
    environment: Environment = Field(default=Environment.LOCAL)
    project_name: str = Field(default="pe-deal-recommender")
    
    # Storage
    storage_backend: StorageBackend = Field(default=StorageBackend.LOCAL)
    compute_backend: ComputeBackend = Field(default=ComputeBackend.LOCAL)
    
    # Paths
    base_dir: Path = Field(default=Path("."))
    data_dir: Path = Field(default=Path("data"))
    models_dir: Path = Field(default=Path("models"))
    logs_dir: Path = Field(default=Path("logs"))
    
    # AWS (placeholder, used when storage_backend=s3)
    aws_region: str = Field(default="us-east-1")
    s3_bucket: str = Field(default="pe-recommender-bucket")
    s3_data_prefix: str = Field(default="data/")
    s3_models_prefix: str = Field(default="models/")
    
    # Ray cluster (used when compute_backend=ray)
    ray_address: Optional[str] = Field(default=None)  # None = start local cluster
    ray_namespace: str = Field(default="pe-recommender")
    
    # ==========================================================================
    # Nested Configs
    # ==========================================================================
    
    data: DataConfig = Field(default_factory=DataConfig)
    two_tower: TwoTowerConfig = Field(default_factory=TwoTowerConfig)
    deep_ranking: DeepRankingConfig = Field(default_factory=DeepRankingConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    ray_training: RayTrainingConfig = Field(default_factory=RayTrainingConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    serving: ServingConfig = Field(default_factory=ServingConfig)
    feature_dims: FeatureDims = Field(default_factory=FeatureDims)
    
    # ==========================================================================
    # Computed Properties
    # ==========================================================================
    
    @property
    def is_local(self) -> bool:
        return self.environment == Environment.LOCAL
    
    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PROD
    
    @property
    def use_ray(self) -> bool:
        return self.compute_backend == ComputeBackend.RAY
    
    @property
    def use_s3(self) -> bool:
        return self.storage_backend == StorageBackend.S3
    
    def get_data_path(self, filename: str) -> Path | str:
        """Get path to a data file, accounting for storage backend."""
        if self.use_s3:
            return f"s3://{self.s3_bucket}/{self.s3_data_prefix}{filename}"
        return self.data_dir / filename
    
    def get_model_path(self, filename: str) -> Path | str:
        """Get path to a model file, accounting for storage backend."""
        if self.use_s3:
            return f"s3://{self.s3_bucket}/{self.s3_models_prefix}{filename}"
        return self.models_dir / filename
    
    # ==========================================================================
    # Initialization
    # ==========================================================================
    
    def ensure_directories(self) -> None:
        """Create necessary directories if they don't exist (local only)."""
        if not self.use_s3:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.models_dir.mkdir(parents=True, exist_ok=True)
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            (self.models_dir / "checkpoints").mkdir(exist_ok=True)
    
    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "Settings":
        """Load settings from a YAML configuration file."""
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, yaml_path: str | Path) -> None:
        """Save settings to a YAML configuration file."""
        with open(yaml_path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)
    
    def __repr__(self) -> str:
        return (
            f"Settings(env={self.environment.value}, "
            f"storage={self.storage_backend.value}, "
            f"compute={self.compute_backend.value})"
        )


# =============================================================================
# Global Settings Instance
# =============================================================================

_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance (lazy initialization)."""
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_directories()
    return _settings


def init_settings(
    config_path: Optional[str | Path] = None,
    **overrides: Any
) -> Settings:
    """
    Initialize settings with optional config file and overrides.
    
    Args:
        config_path: Path to YAML config file
        **overrides: Settings to override
        
    Returns:
        Initialized Settings instance
    """
    global _settings
    
    if config_path:
        _settings = Settings.from_yaml(config_path)
    else:
        _settings = Settings(**overrides)
    
    _settings.ensure_directories()
    return _settings
