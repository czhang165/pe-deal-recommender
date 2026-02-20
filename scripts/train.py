#!/usr/bin/env python3
"""
Main training script for the PE Deal Recommender System.

Usage:
    # Train both models locally
    python scripts/train.py
    
    # Train with custom config
    python scripts/train.py --config config/prod.yaml
    
    # Train only Two-Tower
    python scripts/train.py --model two_tower
    
    # Train with Ray (distributed)
    python scripts/train.py --distributed
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import init_settings, ComputeBackend
from src.training import train_all, train_two_tower, train_deep_ranking


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PE Deal Recommender models"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to YAML config file"
    )
    
    parser.add_argument(
        "--model", "-m",
        type=str,
        choices=["all", "two_tower", "deep_ranking"],
        default="all",
        help="Which model(s) to train"
    )
    
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Use Ray for distributed training"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override max epochs"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override data directory"
    )
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Initialize settings
    overrides = {}
    
    if args.distributed:
        overrides["compute_backend"] = ComputeBackend.RAY
    
    if args.epochs:
        overrides["training"] = {"max_epochs": args.epochs}
    
    if args.batch_size:
        if "training" not in overrides:
            overrides["training"] = {}
        overrides["training"]["batch_size"] = args.batch_size
    
    if args.data_dir:
        overrides["data_dir"] = Path(args.data_dir)
    
    if args.config:
        settings = init_settings(config_path=args.config, **overrides)
    else:
        settings = init_settings(**overrides)
    
    print(f"Settings: {settings}")
    print(f"Data directory: {settings.data_dir}")
    print(f"Models directory: {settings.models_dir}")
    
    # Run training
    if settings.compute_backend == ComputeBackend.RAY:
        print("\n⚠️  Ray distributed training is not fully implemented yet.")
        print("Falling back to local training...\n")
    
    if args.model == "all":
        results = train_all(settings)
        print("\n" + "=" * 60)
        print("Training Results")
        print("=" * 60)
        for name, result in results.items():
            print(f"\n{name}:")
            print(f"  Checkpoint: {result.best_checkpoint_path}")
            print(f"  Best val loss: {result.final_metrics.get('best_val_loss', 'N/A')}")
    
    elif args.model == "two_tower":
        result = train_two_tower(settings=settings)
        print(f"\nTwo-Tower training complete!")
        print(f"Checkpoint: {result.best_checkpoint_path}")
    
    elif args.model == "deep_ranking":
        result = train_deep_ranking(settings=settings)
        print(f"\nDeep Ranking training complete!")
        print(f"Checkpoint: {result.best_checkpoint_path}")


if __name__ == "__main__":
    main()
