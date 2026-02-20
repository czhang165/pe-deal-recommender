"""
PE Deal Recommender System

A production-grade two-stage recommender system for matching investors with
investment opportunities.

Components:
- config: Configuration management with Pydantic
- data: Data loading, preprocessing, and datasets
- models: Two-Tower and Deep Ranking model architectures
- training: Local and distributed training pipelines
- inference: Batch inference with FAISS vector search
- utils: Logging and utilities
"""

__version__ = "1.0.0"

from src.config import Settings, get_settings, init_settings
from src.data import load_data, DataBundle, FeatureEncoder
from src.models import TwoTowerModel, PairwiseDeepRankingModel
from src.training import train_all, train_two_tower, train_deep_ranking
from src.inference import RecommenderInference, load_inference_pipeline

__all__ = [
    # Config
    "Settings",
    "get_settings", 
    "init_settings",
    # Data
    "load_data",
    "DataBundle",
    "FeatureEncoder",
    # Models
    "TwoTowerModel",
    "PairwiseDeepRankingModel",
    # Training
    "train_all",
    "train_two_tower",
    "train_deep_ranking",
    # Inference
    "RecommenderInference",
    "load_inference_pipeline",
]
