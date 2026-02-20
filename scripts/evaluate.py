#!/usr/bin/env python3
"""
Evaluation script for the PE Deal Recommender System.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --k 20
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np

from src.config import init_settings
from src.data import load_data, FeatureEncoder
from src.models import TwoTowerModel, PairwiseDeepRankingModel
from src.training.evaluation import (
    evaluate_model,
    evaluate_two_stage_pipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PE Deal Recommender models"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to YAML config file"
    )
    
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Top-K for evaluation metrics"
    )
    
    parser.add_argument(
        "--retrieval-k",
        type=int,
        default=100,
        help="Number of candidates for retrieval in two-stage"
    )
    
    parser.add_argument(
        "--training-summary",
        type=str,
        default=None,
        help="Path to training_summary.json (auto-detected if not provided)"
    )
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Initialize settings
    if args.config:
        settings = init_settings(config_path=args.config)
    else:
        settings = init_settings()
    
    # Load training summary
    summary_path = args.training_summary or settings.models_dir / "training_summary.json"
    
    if not Path(summary_path).exists():
        print(f"Error: Training summary not found at {summary_path}")
        print("Run training first with: python scripts/train.py")
        sys.exit(1)
    
    with open(summary_path, "r") as f:
        summary = json.load(f)
    
    print("Loading data...")
    data_bundle = load_data(settings)
    
    print("Loading models...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load Two-Tower model
    two_tower_ckpt = summary["two_tower"]["best_checkpoint_path"]
    two_tower_model = TwoTowerModel.load_from_checkpoint(
        two_tower_ckpt,
        n_investors=data_bundle.n_investors,
        n_deals=data_bundle.n_deals,
        feature_dims=data_bundle.encoder.feature_dims,
    )
    two_tower_model = two_tower_model.to(device)
    two_tower_model.eval()
    
    # Load Deep Ranking model
    deep_ranking_ckpt = summary["deep_ranking"]["best_checkpoint_path"]
    deep_ranking_model = PairwiseDeepRankingModel.load_from_checkpoint(
        deep_ranking_ckpt,
        n_investors=data_bundle.n_investors,
        n_deals=data_bundle.n_deals,
        feature_dims=data_bundle.encoder.feature_dims,
    )
    deep_ranking_model = deep_ranking_model.to(device)
    deep_ranking_model.eval()
    
    print(f"\nEvaluating with K={args.k}, retrieval_k={args.retrieval_k}")
    print("=" * 60)
    
    all_deal_ids = data_bundle.all_deal_ids
    
    # Evaluate Two-Tower alone
    print("\n1. Evaluating Two-Tower Model...")
    two_tower_metrics = evaluate_model(
        model=two_tower_model,
        test_interactions=data_bundle.test_interactions,
        train_interactions=data_bundle.train_interactions,
        investor_features=data_bundle.investor_features,
        deal_features=data_bundle.deal_features,
        all_deal_ids=all_deal_ids,
        k=args.k,
    )
    print(f"   {two_tower_metrics}")
    
    # Evaluate Deep Ranking alone
    print("\n2. Evaluating Deep Ranking Model...")
    deep_ranking_metrics = evaluate_model(
        model=deep_ranking_model,
        test_interactions=data_bundle.test_interactions,
        train_interactions=data_bundle.train_interactions,
        investor_features=data_bundle.investor_features,
        deal_features=data_bundle.deal_features,
        all_deal_ids=all_deal_ids,
        k=args.k,
    )
    print(f"   {deep_ranking_metrics}")
    
    # Evaluate Two-Stage Pipeline
    print("\n3. Evaluating Two-Stage Pipeline...")
    two_stage_metrics = evaluate_two_stage_pipeline(
        retrieval_model=two_tower_model,
        ranking_model=deep_ranking_model,
        test_interactions=data_bundle.test_interactions,
        train_interactions=data_bundle.train_interactions,
        investor_features=data_bundle.investor_features,
        deal_features=data_bundle.deal_features,
        all_deal_ids=all_deal_ids,
        retrieval_k=args.retrieval_k,
        final_k=args.k,
    )
    print(f"   {two_stage_metrics}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\n{'Model':<25} {'Hit@K':<10} {'NDCG@K':<10} {'MRR':<10}")
    print("-" * 55)
    print(f"{'Two-Tower':<25} {two_tower_metrics.hit_at_k:<10.4f} {two_tower_metrics.ndcg_at_k:<10.4f} {two_tower_metrics.mrr:<10.4f}")
    print(f"{'Deep Ranking':<25} {deep_ranking_metrics.hit_at_k:<10.4f} {deep_ranking_metrics.ndcg_at_k:<10.4f} {deep_ranking_metrics.mrr:<10.4f}")
    print(f"{'Two-Stage Pipeline':<25} {two_stage_metrics.hit_at_k:<10.4f} {two_stage_metrics.ndcg_at_k:<10.4f} {two_stage_metrics.mrr:<10.4f}")
    
    # Save results
    results = {
        "k": args.k,
        "retrieval_k": args.retrieval_k,
        "two_tower": two_tower_metrics.to_dict(),
        "deep_ranking": deep_ranking_metrics.to_dict(),
        "two_stage": two_stage_metrics.to_dict(),
    }
    
    results_path = settings.models_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
