"""
Evaluation metrics and utilities for the recommender system.

Includes:
- Hit@K: Whether any relevant item is in top-K
- NDCG@K: Normalized Discounted Cumulative Gain
- MRR: Mean Reciprocal Rank
- Recall@K: Fraction of relevant items in top-K
- Precision@K: Fraction of top-K that are relevant
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.config import Settings, get_settings
from src.data import DataBundle


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    
    hit_at_k: float = 0.0
    ndcg_at_k: float = 0.0
    mrr: float = 0.0
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    k: int = 10
    
    # Per-investor results for analysis
    per_investor_results: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            f"hit@{self.k}": self.hit_at_k,
            f"ndcg@{self.k}": self.ndcg_at_k,
            "mrr": self.mrr,
            f"recall@{self.k}": self.recall_at_k,
            f"precision@{self.k}": self.precision_at_k,
        }
    
    def __repr__(self) -> str:
        return (
            f"EvaluationMetrics(hit@{self.k}={self.hit_at_k:.4f}, "
            f"ndcg@{self.k}={self.ndcg_at_k:.4f}, "
            f"mrr={self.mrr:.4f}, "
            f"recall@{self.k}={self.recall_at_k:.4f}, "
            f"precision@{self.k}={self.precision_at_k:.4f})"
        )


def compute_ndcg(ranked_items: list[int], relevant_items: set[int], k: int) -> float:
    """
    Compute Normalized Discounted Cumulative Gain.
    
    Uses binary relevance (1 if relevant, 0 otherwise).
    """
    dcg = 0.0
    for i, item in enumerate(ranked_items[:k]):
        if item in relevant_items:
            dcg += 1.0 / np.log2(i + 2)  # i+2 because position starts at 1
    
    # Ideal DCG: all relevant items at top
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant_items), k)))
    
    return dcg / idcg if idcg > 0 else 0.0


def compute_mrr(ranked_items: list[int], relevant_items: set[int]) -> float:
    """
    Compute Mean Reciprocal Rank.
    
    Returns 1/rank of first relevant item found.
    """
    for rank, item in enumerate(ranked_items, start=1):
        if item in relevant_items:
            return 1.0 / rank
    return 0.0


def score_deals_for_investor(
    model: torch.nn.Module,
    investor_id: int,
    deal_ids: np.ndarray,
    investor_features: dict[str, np.ndarray],
    deal_features: dict[str, np.ndarray],
    batch_size: int = 256,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Score all specified deals for a given investor.
    
    Args:
        model: Trained model (TwoTowerModel or PairwiseDeepRankingModel)
        investor_id: ID of the investor
        deal_ids: Array of deal IDs to score
        investor_features: Dict of investor features
        deal_features: Dict of deal features
        batch_size: Batch size for scoring
        device: Device to use for computation
        
    Returns:
        Array of scores for each deal
    """
    model.eval()
    
    if device is None:
        device = next(model.parameters()).device
    
    all_scores = []
    n_deals = len(deal_ids)
    
    # Get investor features once
    inv_type = investor_features["type"][investor_id]
    inv_region = investor_features["region"][investor_id]
    inv_risk = investor_features["risk"][investor_id]
    inv_min = investor_features["min_investment"][investor_id]
    inv_max = investor_features["max_investment"][investor_id]
    inv_exp = investor_features["experience_years"][investor_id]
    inv_port = investor_features["portfolio_size"][investor_id]
    
    with torch.no_grad():
        for start_idx in range(0, n_deals, batch_size):
            end_idx = min(start_idx + batch_size, n_deals)
            batch_deals = deal_ids[start_idx:end_idx]
            batch_size_actual = len(batch_deals)
            
            # Prepare investor data (repeated for each deal)
            investor_data = {
                "id": torch.tensor([investor_id] * batch_size_actual, device=device),
                "type": torch.tensor([inv_type] * batch_size_actual, device=device),
                "region": torch.tensor([inv_region] * batch_size_actual, device=device),
                "risk": torch.tensor([inv_risk] * batch_size_actual, device=device),
                "min_investment": torch.tensor(
                    [inv_min] * batch_size_actual, dtype=torch.float32, device=device
                ),
                "max_investment": torch.tensor(
                    [inv_max] * batch_size_actual, dtype=torch.float32, device=device
                ),
                "experience_years": torch.tensor(
                    [inv_exp] * batch_size_actual, dtype=torch.float32, device=device
                ),
                "portfolio_size": torch.tensor(
                    [inv_port] * batch_size_actual, dtype=torch.float32, device=device
                ),
            }
            
            # Prepare deal data
            deal_data = {
                "id": torch.tensor(batch_deals, device=device),
                "sector": torch.tensor(
                    [deal_features["sector"][d] for d in batch_deals], device=device
                ),
                "stage": torch.tensor(
                    [deal_features["stage"][d] for d in batch_deals], device=device
                ),
                "region": torch.tensor(
                    [deal_features["region"][d] for d in batch_deals], device=device
                ),
                "deal_size": torch.tensor(
                    [deal_features["deal_size"][d] for d in batch_deals],
                    dtype=torch.float32, device=device
                ),
                "revenue_multiple": torch.tensor(
                    [deal_features["revenue_multiple"][d] for d in batch_deals],
                    dtype=torch.float32, device=device
                ),
                "growth_rate": torch.tensor(
                    [deal_features["growth_rate"][d] for d in batch_deals],
                    dtype=torch.float32, device=device
                ),
                "profitability": torch.tensor(
                    [deal_features["profitability"][d] for d in batch_deals],
                    dtype=torch.float32, device=device
                ),
                "team_experience": torch.tensor(
                    [deal_features["team_experience"][d] for d in batch_deals],
                    dtype=torch.float32, device=device
                ),
                "market_size": torch.tensor(
                    [deal_features["market_size"][d] for d in batch_deals],
                    dtype=torch.float32, device=device
                ),
            }
            
            # Get scores
            scores = model(investor_data, deal_data)
            if scores.dim() > 1:
                scores = scores.squeeze(-1)
            
            all_scores.append(scores.cpu().numpy())
    
    return np.concatenate(all_scores)


def evaluate_model(
    model: torch.nn.Module,
    test_interactions: pd.DataFrame,
    train_interactions: pd.DataFrame,
    investor_features: dict[str, np.ndarray],
    deal_features: dict[str, np.ndarray],
    all_deal_ids: np.ndarray,
    k: int = 10,
    show_progress: bool = True,
) -> EvaluationMetrics:
    """
    Evaluate a model on test interactions.
    
    Excludes training positives from candidates to ensure fair evaluation.
    
    Args:
        model: Trained model
        test_interactions: Test interaction DataFrame
        train_interactions: Training interaction DataFrame
        investor_features: Dict of investor features
        deal_features: Dict of deal features
        all_deal_ids: Array of all deal IDs
        k: Top-K for metrics
        show_progress: Whether to show progress bar
        
    Returns:
        EvaluationMetrics with all computed metrics
    """
    results = {
        "hit": [],
        "ndcg": [],
        "mrr": [],
        "recall": [],
        "precision": [],
    }
    
    # Build sets of positive deals per investor
    train_positives: dict[int, set[int]] = {}
    for _, row in train_interactions.iterrows():
        inv_id = int(row["investorId"])
        deal_id = int(row["dealId"])
        if inv_id not in train_positives:
            train_positives[inv_id] = set()
        train_positives[inv_id].add(deal_id)
    
    # Group test interactions by investor
    grouped = test_interactions.groupby("investorId")
    
    iterator = grouped
    if show_progress:
        iterator = tqdm(grouped, desc="Evaluating")
    
    for investor_id, group in iterator:
        investor_id = int(investor_id)
        
        # Get test positive deals for this investor
        test_positive_deals = set(group["dealId"].values.astype(int))
        
        # Get training positive deals for this investor
        train_positive_deals = train_positives.get(investor_id, set())
        
        # Candidate deals = all deals - training positives
        candidate_deals = np.array([
            d for d in all_deal_ids if d not in train_positive_deals
        ])
        
        if len(candidate_deals) == 0:
            continue
        
        # Score all candidates
        scores = score_deals_for_investor(
            model, investor_id, candidate_deals,
            investor_features, deal_features
        )
        
        # Rank by score (descending)
        ranked_indices = np.argsort(-scores)
        ranked_deals = candidate_deals[ranked_indices].tolist()
        top_k_deals = ranked_deals[:k]
        
        # Compute metrics
        hit = int(any(d in test_positive_deals for d in top_k_deals))
        ndcg = compute_ndcg(ranked_deals, test_positive_deals, k)
        mrr = compute_mrr(ranked_deals, test_positive_deals)
        
        n_relevant_in_k = len(set(top_k_deals) & test_positive_deals)
        recall = n_relevant_in_k / len(test_positive_deals) if test_positive_deals else 0
        precision = n_relevant_in_k / k
        
        results["hit"].append(hit)
        results["ndcg"].append(ndcg)
        results["mrr"].append(mrr)
        results["recall"].append(recall)
        results["precision"].append(precision)
    
    return EvaluationMetrics(
        hit_at_k=np.mean(results["hit"]) if results["hit"] else 0.0,
        ndcg_at_k=np.mean(results["ndcg"]) if results["ndcg"] else 0.0,
        mrr=np.mean(results["mrr"]) if results["mrr"] else 0.0,
        recall_at_k=np.mean(results["recall"]) if results["recall"] else 0.0,
        precision_at_k=np.mean(results["precision"]) if results["precision"] else 0.0,
        k=k,
    )


def evaluate_two_stage_pipeline(
    retrieval_model: torch.nn.Module,
    ranking_model: torch.nn.Module,
    test_interactions: pd.DataFrame,
    train_interactions: pd.DataFrame,
    investor_features: dict[str, np.ndarray],
    deal_features: dict[str, np.ndarray],
    all_deal_ids: np.ndarray,
    retrieval_k: int = 100,
    final_k: int = 10,
    show_progress: bool = True,
) -> EvaluationMetrics:
    """
    Evaluate the two-stage retrieval + ranking pipeline.
    
    1. Use retrieval model to get top retrieval_k candidates
    2. Use ranking model to re-rank and select top final_k
    
    Args:
        retrieval_model: Two-Tower model for retrieval
        ranking_model: Deep Ranking model for re-ranking
        test_interactions: Test interaction DataFrame
        train_interactions: Training interaction DataFrame
        investor_features: Dict of investor features
        deal_features: Dict of deal features
        all_deal_ids: Array of all deal IDs
        retrieval_k: Number of candidates to retrieve
        final_k: Final top-K after re-ranking
        show_progress: Whether to show progress bar
        
    Returns:
        EvaluationMetrics for the two-stage pipeline
    """
    results = {
        "hit": [],
        "ndcg": [],
        "mrr": [],
        "recall": [],
        "precision": [],
    }
    
    # Build training positives
    train_positives: dict[int, set[int]] = {}
    for _, row in train_interactions.iterrows():
        inv_id = int(row["investorId"])
        deal_id = int(row["dealId"])
        if inv_id not in train_positives:
            train_positives[inv_id] = set()
        train_positives[inv_id].add(deal_id)
    
    grouped = test_interactions.groupby("investorId")
    
    iterator = grouped
    if show_progress:
        iterator = tqdm(grouped, desc="Evaluating two-stage pipeline")
    
    for investor_id, group in iterator:
        investor_id = int(investor_id)
        test_positive_deals = set(group["dealId"].values.astype(int))
        train_positive_deals = train_positives.get(investor_id, set())
        
        # Candidates = all - training positives
        candidate_deals = np.array([
            d for d in all_deal_ids if d not in train_positive_deals
        ])
        
        if len(candidate_deals) == 0:
            continue
        
        # Stage 1: Retrieval
        retrieval_scores = score_deals_for_investor(
            retrieval_model, investor_id, candidate_deals,
            investor_features, deal_features
        )
        retrieval_indices = np.argsort(-retrieval_scores)[:retrieval_k]
        retrieved_deals = candidate_deals[retrieval_indices]
        
        # Stage 2: Re-ranking
        if len(retrieved_deals) > 0:
            ranking_scores = score_deals_for_investor(
                ranking_model, investor_id, retrieved_deals,
                investor_features, deal_features
            )
            final_indices = np.argsort(-ranking_scores)[:final_k]
            final_deals = retrieved_deals[final_indices].tolist()
        else:
            final_deals = []
        
        # Compute metrics
        top_k_deals = final_deals[:final_k]
        
        hit = int(any(d in test_positive_deals for d in top_k_deals))
        ndcg = compute_ndcg(final_deals, test_positive_deals, final_k)
        mrr = compute_mrr(final_deals, test_positive_deals)
        
        n_relevant_in_k = len(set(top_k_deals) & test_positive_deals)
        recall = n_relevant_in_k / len(test_positive_deals) if test_positive_deals else 0
        precision = n_relevant_in_k / final_k if final_k > 0 else 0
        
        results["hit"].append(hit)
        results["ndcg"].append(ndcg)
        results["mrr"].append(mrr)
        results["recall"].append(recall)
        results["precision"].append(precision)
    
    return EvaluationMetrics(
        hit_at_k=np.mean(results["hit"]) if results["hit"] else 0.0,
        ndcg_at_k=np.mean(results["ndcg"]) if results["ndcg"] else 0.0,
        mrr=np.mean(results["mrr"]) if results["mrr"] else 0.0,
        recall_at_k=np.mean(results["recall"]) if results["recall"] else 0.0,
        precision_at_k=np.mean(results["precision"]) if results["precision"] else 0.0,
        k=final_k,
    )
