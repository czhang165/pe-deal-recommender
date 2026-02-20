"""
Data loading utilities for the recommender system.

Handles:
- Loading data from local files or S3
- Train/val/test splitting
- Creating PyTorch DataLoaders
- Optional Ray Data integration for distributed training
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import Settings, get_settings
from src.data.datasets import (
    InvestorDealDataset,
    PairwiseRankingDataset,
    collate_fn,
    pairwise_collate_fn,
)
from src.data.feature_encoder import FeatureEncoder


@dataclass
class DataBundle:
    """
    Container for all data needed for training and evaluation.
    
    Attributes:
        interactions_df: All interactions
        train_interactions: Training split
        val_interactions: Validation split
        test_interactions: Test split
        investor_df: Investor features DataFrame
        deal_df: Deal features DataFrame
        encoder: Fitted FeatureEncoder
        investor_features: Encoded investor features (dict of numpy arrays)
        deal_features: Encoded deal features (dict of numpy arrays)
    """
    interactions_df: pd.DataFrame
    train_interactions: pd.DataFrame
    val_interactions: pd.DataFrame
    test_interactions: pd.DataFrame
    investor_df: pd.DataFrame
    deal_df: pd.DataFrame
    encoder: FeatureEncoder
    investor_features: dict[str, np.ndarray]
    deal_features: dict[str, np.ndarray]
    
    @property
    def n_investors(self) -> int:
        return len(self.investor_df)
    
    @property
    def n_deals(self) -> int:
        return len(self.deal_df)
    
    @property
    def all_deal_ids(self) -> np.ndarray:
        return self.deal_df.index.values


def load_data(settings: Optional[Settings] = None) -> DataBundle:
    """
    Load and prepare all data for training.
    
    Args:
        settings: Configuration settings (uses global if not provided)
        
    Returns:
        DataBundle containing all prepared data
    """
    if settings is None:
        settings = get_settings()
    
    # Load raw data
    interactions_df = _load_interactions(settings)
    investor_df = _load_investor_features(settings)
    deal_df = _load_deal_features(settings)
    
    # Split data
    train_interactions, val_interactions, test_interactions = _split_data(
        interactions_df, settings
    )
    
    # Fit feature encoder
    encoder = FeatureEncoder()
    encoder.fit(investor_df, deal_df)
    
    # Update settings with actual feature dims
    settings.feature_dims = encoder.feature_dims
    
    # Encode features
    investor_features = encoder.encode_investor_features(investor_df)
    deal_features = encoder.encode_deal_features(deal_df)
    
    return DataBundle(
        interactions_df=interactions_df,
        train_interactions=train_interactions,
        val_interactions=val_interactions,
        test_interactions=test_interactions,
        investor_df=investor_df,
        deal_df=deal_df,
        encoder=encoder,
        investor_features=investor_features,
        deal_features=deal_features,
    )


def _load_interactions(settings: Settings) -> pd.DataFrame:
    """Load interactions data."""
    path = settings.get_data_path(settings.data.interactions_file)
    
    if settings.use_s3:
        df = pd.read_csv(path, storage_options=_get_s3_options(settings))
    else:
        df = pd.read_csv(path)
    
    # Parse timestamp if present
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    return df


def _load_investor_features(settings: Settings) -> pd.DataFrame:
    """Load investor features data."""
    path = settings.get_data_path(settings.data.investor_features_file)
    
    if settings.use_s3:
        df = pd.read_csv(path, storage_options=_get_s3_options(settings))
    else:
        df = pd.read_csv(path)
    
    # Ensure investorId is index for fast lookup
    if "investorId" in df.columns:
        df = df.set_index("investorId")
    
    return df


def _load_deal_features(settings: Settings) -> pd.DataFrame:
    """Load deal features data."""
    path = settings.get_data_path(settings.data.deal_features_file)
    
    if settings.use_s3:
        df = pd.read_csv(path, storage_options=_get_s3_options(settings))
    else:
        df = pd.read_csv(path)
    
    # Ensure dealId is index for fast lookup
    if "dealId" in df.columns:
        df = df.set_index("dealId")
    
    return df


def _get_s3_options(settings: Settings) -> dict:
    """Get S3 storage options for pandas."""
    return {
        "key": None,  # Use default credentials
        "secret": None,
        "client_kwargs": {"region_name": settings.aws_region},
    }


def _split_data(
    interactions_df: pd.DataFrame, 
    settings: Settings
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split interactions into train/val/test sets.
    
    Uses temporal splitting if timestamp is available, otherwise random split.
    """
    n_total = len(interactions_df)
    n_test = int(n_total * settings.data.test_size)
    n_val = int(n_total * settings.data.val_size)
    n_train = n_total - n_test - n_val
    
    if "timestamp" in interactions_df.columns:
        # Temporal split: sort by time and take most recent for test
        interactions_df = interactions_df.sort_values("timestamp")
        train = interactions_df.iloc[:n_train]
        val = interactions_df.iloc[n_train:n_train + n_val]
        test = interactions_df.iloc[n_train + n_val:]
    else:
        # Random split
        rng = np.random.default_rng(42)
        indices = rng.permutation(n_total)
        train = interactions_df.iloc[indices[:n_train]]
        val = interactions_df.iloc[indices[n_train:n_train + n_val]]
        test = interactions_df.iloc[indices[n_train + n_val:]]
    
    return train, val, test


def create_dataloaders(
    data_bundle: DataBundle,
    settings: Optional[Settings] = None,
    model_type: str = "two_tower",
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch DataLoaders for training.
    
    Args:
        data_bundle: Prepared data bundle
        settings: Configuration settings
        model_type: "two_tower" or "deep_ranking"
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    if settings is None:
        settings = get_settings()
    
    if model_type == "two_tower":
        return _create_pointwise_dataloaders(data_bundle, settings)
    elif model_type == "deep_ranking":
        return _create_pairwise_dataloaders(data_bundle, settings)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def _create_pointwise_dataloaders(
    data_bundle: DataBundle,
    settings: Settings,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for pointwise (Two-Tower) training."""
    
    train_dataset = InvestorDealDataset(
        interactions_df=data_bundle.train_interactions,
        investor_features=data_bundle.investor_features,
        deal_features=data_bundle.deal_features,
        all_deal_ids=data_bundle.all_deal_ids,
        negative_samples=settings.data.negative_samples,
    )
    
    val_dataset = InvestorDealDataset(
        interactions_df=data_bundle.val_interactions,
        investor_features=data_bundle.investor_features,
        deal_features=data_bundle.deal_features,
        all_deal_ids=data_bundle.all_deal_ids,
        negative_samples=settings.data.negative_samples,
        seed=43,  # Different seed for val
    )
    
    test_dataset = InvestorDealDataset(
        interactions_df=data_bundle.test_interactions,
        investor_features=data_bundle.investor_features,
        deal_features=data_bundle.deal_features,
        all_deal_ids=data_bundle.all_deal_ids,
        negative_samples=settings.data.negative_samples,
        seed=44,  # Different seed for test
    )
    
    # Determine number of workers
    num_workers = settings.training.num_workers
    if not torch.cuda.is_available():
        num_workers = min(num_workers, 2)  # Reduce for CPU-only
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=settings.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=settings.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=settings.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )
    
    return train_loader, val_loader, test_loader


def _create_pairwise_dataloaders(
    data_bundle: DataBundle,
    settings: Settings,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders for pairwise (Deep Ranking) training."""
    
    train_dataset = PairwiseRankingDataset(
        interactions_df=data_bundle.train_interactions,
        investor_features=data_bundle.investor_features,
        deal_features=data_bundle.deal_features,
        n_deals=data_bundle.n_deals,
        n_pairs_per_positive=settings.data.pairwise_pairs_per_positive,
        show_progress=True,
    )
    
    val_dataset = PairwiseRankingDataset(
        interactions_df=data_bundle.val_interactions,
        investor_features=data_bundle.investor_features,
        deal_features=data_bundle.deal_features,
        n_deals=data_bundle.n_deals,
        n_pairs_per_positive=3,  # Fewer pairs for validation
        seed=43,
        show_progress=False,
    )
    
    test_dataset = PairwiseRankingDataset(
        interactions_df=data_bundle.test_interactions,
        investor_features=data_bundle.investor_features,
        deal_features=data_bundle.deal_features,
        n_deals=data_bundle.n_deals,
        n_pairs_per_positive=3,
        seed=44,
        show_progress=False,
    )
    
    # Determine number of workers
    num_workers = settings.training.num_workers
    if not torch.cuda.is_available():
        num_workers = min(num_workers, 2)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=settings.training.batch_size,
        shuffle=True,
        collate_fn=pairwise_collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=settings.training.batch_size,
        shuffle=False,
        collate_fn=pairwise_collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=settings.training.batch_size,
        shuffle=False,
        collate_fn=pairwise_collate_fn,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )
    
    return train_loader, val_loader, test_loader
