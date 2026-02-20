"""Training module for the recommender system."""

from src.training.local_trainer import (
    TrainingResult,
    train_two_tower,
    train_deep_ranking,
    train_all,
)
from src.training.evaluation import (
    EvaluationMetrics,
    evaluate_model,
    evaluate_two_stage_pipeline,
    score_deals_for_investor,
)

__all__ = [
    "TrainingResult",
    "train_two_tower",
    "train_deep_ranking",
    "train_all",
    "EvaluationMetrics",
    "evaluate_model",
    "evaluate_two_stage_pipeline",
    "score_deals_for_investor",
]

# Optional Ray training imports
try:
    from src.training.ray_trainer import (
        init_ray,
        train_two_tower_distributed,
        train_deep_ranking_distributed,
        shutdown_ray,
    )
    __all__.extend([
        "init_ray",
        "train_two_tower_distributed", 
        "train_deep_ranking_distributed",
        "shutdown_ray",
    ])
except ImportError:
    pass  # Ray not installed
