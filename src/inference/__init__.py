"""Inference module for the recommender system."""

from src.inference.batch_inference import (
    RecommendationResult,
    FAISSIndex,
    RecommenderInference,
    load_inference_pipeline,
)

__all__ = [
    "RecommendationResult",
    "FAISSIndex",
    "RecommenderInference",
    "load_inference_pipeline",
]
