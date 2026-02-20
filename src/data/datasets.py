"""
PyTorch Dataset classes for investor-deal recommendation.

Includes:
- InvestorDealDataset: Pointwise dataset with negative sampling
- PairwiseRankingDataset: Pairwise dataset for RankNet-style training
- Collate functions for batching
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class InvestorDealDataset(Dataset):
    """
    Dataset for pointwise training with negative sampling.
    
    Each sample is a (investor, deal, label) tuple where:
    - Positive samples (label=1): actual interactions
    - Negative samples (label=0): randomly sampled non-interactions
    
    Args:
        interactions_df: DataFrame with columns [investorId, dealId, ...]
        investor_features: Dict of investor features keyed by investorId
        deal_features: Dict of deal features keyed by dealId
        all_deal_ids: Array of all valid deal IDs
        negative_samples: Number of negative samples per positive
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        interactions_df: pd.DataFrame,
        investor_features: dict[str, np.ndarray],
        deal_features: dict[str, np.ndarray],
        all_deal_ids: np.ndarray,
        negative_samples: int = 4,
        seed: int = 42,
    ):
        self.investor_features = investor_features
        self.deal_features = deal_features
        self.all_deal_ids = set(all_deal_ids)
        self.negative_samples = negative_samples
        self.rng = np.random.default_rng(seed)
        
        # Build positive interactions
        self.positive_pairs: list[tuple[int, int]] = []
        self.investor_positive_deals: dict[int, set[int]] = {}
        
        for _, row in interactions_df.iterrows():
            inv_id = int(row["investorId"])
            deal_id = int(row["dealId"])
            
            self.positive_pairs.append((inv_id, deal_id))
            
            if inv_id not in self.investor_positive_deals:
                self.investor_positive_deals[inv_id] = set()
            self.investor_positive_deals[inv_id].add(deal_id)
        
        # Pre-compute negative candidates for each investor
        self.investor_negative_candidates: dict[int, np.ndarray] = {}
        for inv_id, positive_deals in self.investor_positive_deals.items():
            negative_deals = list(self.all_deal_ids - positive_deals)
            self.investor_negative_candidates[inv_id] = np.array(negative_deals)
    
    def __len__(self) -> int:
        return len(self.positive_pairs) * (1 + self.negative_samples)
    
    def __getitem__(self, idx: int) -> tuple[dict[str, Any], dict[str, Any], float]:
        n_positives = len(self.positive_pairs)
        
        if idx < n_positives:
            # Positive sample
            investor_id, deal_id = self.positive_pairs[idx]
            label = 1.0
        else:
            # Negative sample
            pos_idx = (idx - n_positives) // self.negative_samples
            investor_id, _ = self.positive_pairs[pos_idx]
            
            # Sample negative deal
            negative_candidates = self.investor_negative_candidates[investor_id]
            deal_id = int(self.rng.choice(negative_candidates))
            label = 0.0
        
        # Get features
        investor_data = self._get_investor_data(investor_id)
        deal_data = self._get_deal_data(deal_id)
        
        return investor_data, deal_data, label
    
    def _get_investor_data(self, investor_id: int) -> dict[str, Any]:
        """Get investor features for a single investor."""
        return {
            "id": investor_id,
            "type": int(self.investor_features["type"][investor_id]),
            "region": int(self.investor_features["region"][investor_id]),
            "risk": int(self.investor_features["risk"][investor_id]),
            "min_investment": float(self.investor_features["min_investment"][investor_id]),
            "max_investment": float(self.investor_features["max_investment"][investor_id]),
            "experience_years": float(self.investor_features["experience_years"][investor_id]),
            "portfolio_size": float(self.investor_features["portfolio_size"][investor_id]),
        }
    
    def _get_deal_data(self, deal_id: int) -> dict[str, Any]:
        """Get deal features for a single deal."""
        return {
            "id": deal_id,
            "sector": int(self.deal_features["sector"][deal_id]),
            "stage": int(self.deal_features["stage"][deal_id]),
            "region": int(self.deal_features["region"][deal_id]),
            "deal_size": float(self.deal_features["deal_size"][deal_id]),
            "revenue_multiple": float(self.deal_features["revenue_multiple"][deal_id]),
            "growth_rate": float(self.deal_features["growth_rate"][deal_id]),
            "profitability": float(self.deal_features["profitability"][deal_id]),
            "team_experience": float(self.deal_features["team_experience"][deal_id]),
            "market_size": float(self.deal_features["market_size"][deal_id]),
        }


class PairwiseRankingDataset(Dataset):
    """
    Dataset for pairwise ranking (RankNet-style training).
    
    Each sample is a (investor, better_deal, worse_deal) tuple where:
    - better_deal: a deal the investor actually interacted with
    - worse_deal: a randomly sampled deal the investor did NOT interact with
    
    The model should learn to score better_deal > worse_deal.
    
    Args:
        interactions_df: DataFrame with columns [investorId, dealId, ...]
        investor_features: Dict of investor features keyed by investorId
        deal_features: Dict of deal features keyed by dealId
        n_deals: Total number of deals
        n_pairs_per_positive: Number of negative pairs to create per positive interaction
        seed: Random seed for reproducibility
        show_progress: Whether to show progress bar during pair generation
    """
    
    def __init__(
        self,
        interactions_df: pd.DataFrame,
        investor_features: dict[str, np.ndarray],
        deal_features: dict[str, np.ndarray],
        n_deals: int,
        n_pairs_per_positive: int = 5,
        seed: int = 42,
        show_progress: bool = True,
    ):
        self.investor_features = investor_features
        self.deal_features = deal_features
        self.pairs: list[dict[str, int]] = []
        
        rng = np.random.default_rng(seed)
        
        # Build investor -> positive deals mapping
        investor_positive_deals: dict[int, set[int]] = {}
        for _, row in interactions_df.iterrows():
            inv_id = int(row["investorId"])
            deal_id = int(row["dealId"])
            if inv_id not in investor_positive_deals:
                investor_positive_deals[inv_id] = set()
            investor_positive_deals[inv_id].add(deal_id)
        
        # All deal IDs
        all_deals = set(range(n_deals))
        
        # Generate pairs
        iterator = investor_positive_deals.items()
        if show_progress:
            iterator = tqdm(iterator, desc="Generating pairwise training data")
        
        for investor_id, positive_deals in iterator:
            negative_deals = list(all_deals - positive_deals)
            
            if len(negative_deals) == 0:
                continue
            
            for pos_deal in positive_deals:
                n_samples = min(n_pairs_per_positive, len(negative_deals))
                sampled_negatives = rng.choice(
                    negative_deals, size=n_samples, replace=False
                )
                
                for neg_deal in sampled_negatives:
                    self.pairs.append({
                        "investor_id": investor_id,
                        "better_deal_id": pos_deal,
                        "worse_deal_id": int(neg_deal),
                    })
        
        if show_progress:
            print(f"Created {len(self.pairs)} training pairs")
    
    def __len__(self) -> int:
        return len(self.pairs)
    
    def __getitem__(
        self, idx: int
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        pair = self.pairs[idx]
        investor_id = pair["investor_id"]
        better_deal_id = pair["better_deal_id"]
        worse_deal_id = pair["worse_deal_id"]
        
        investor_data = self._get_investor_data(investor_id)
        better_deal_data = self._get_deal_data(better_deal_id)
        worse_deal_data = self._get_deal_data(worse_deal_id)
        
        return investor_data, better_deal_data, worse_deal_data
    
    def _get_investor_data(self, investor_id: int) -> dict[str, Any]:
        """Get investor features for a single investor."""
        return {
            "id": investor_id,
            "type": int(self.investor_features["type"][investor_id]),
            "region": int(self.investor_features["region"][investor_id]),
            "risk": int(self.investor_features["risk"][investor_id]),
            "min_investment": float(self.investor_features["min_investment"][investor_id]),
            "max_investment": float(self.investor_features["max_investment"][investor_id]),
            "experience_years": float(self.investor_features["experience_years"][investor_id]),
            "portfolio_size": float(self.investor_features["portfolio_size"][investor_id]),
        }
    
    def _get_deal_data(self, deal_id: int) -> dict[str, Any]:
        """Get deal features for a single deal."""
        return {
            "id": deal_id,
            "sector": int(self.deal_features["sector"][deal_id]),
            "stage": int(self.deal_features["stage"][deal_id]),
            "region": int(self.deal_features["region"][deal_id]),
            "deal_size": float(self.deal_features["deal_size"][deal_id]),
            "revenue_multiple": float(self.deal_features["revenue_multiple"][deal_id]),
            "growth_rate": float(self.deal_features["growth_rate"][deal_id]),
            "profitability": float(self.deal_features["profitability"][deal_id]),
            "team_experience": float(self.deal_features["team_experience"][deal_id]),
            "market_size": float(self.deal_features["market_size"][deal_id]),
        }


def collate_fn(
    batch: list[tuple[dict, dict, float]]
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
    """
    Custom collate function for pointwise data.
    
    Converts list of (investor_data, deal_data, label) tuples into
    batched tensors.
    """
    investor_data: dict[str, list] = {}
    deal_data: dict[str, list] = {}
    labels: list[float] = []
    
    for inv_data, d_data, label in batch:
        for key, value in inv_data.items():
            if key not in investor_data:
                investor_data[key] = []
            investor_data[key].append(value)
        
        for key, value in d_data.items():
            if key not in deal_data:
                deal_data[key] = []
            deal_data[key].append(value)
        
        labels.append(label)
    
    # Convert to tensors
    investor_tensors = {}
    for key, values in investor_data.items():
        if key in ["min_investment", "max_investment", "experience_years", "portfolio_size"]:
            investor_tensors[key] = torch.tensor(values, dtype=torch.float32)
        else:
            investor_tensors[key] = torch.tensor(values, dtype=torch.long)
    
    deal_tensors = {}
    for key, values in deal_data.items():
        if key in ["deal_size", "revenue_multiple", "growth_rate", 
                   "profitability", "team_experience", "market_size"]:
            deal_tensors[key] = torch.tensor(values, dtype=torch.float32)
        else:
            deal_tensors[key] = torch.tensor(values, dtype=torch.long)
    
    labels_tensor = torch.tensor(labels, dtype=torch.float32)
    
    return investor_tensors, deal_tensors, labels_tensor


def pairwise_collate_fn(
    batch: list[tuple[dict, dict, dict]]
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """
    Custom collate function for pairwise ranking data.
    
    Converts list of (investor_data, better_deal_data, worse_deal_data) tuples
    into batched tensors.
    """
    investor_data: dict[str, list] = {}
    better_deal_data: dict[str, list] = {}
    worse_deal_data: dict[str, list] = {}
    
    for inv_data, better_data, worse_data in batch:
        for key, value in inv_data.items():
            if key not in investor_data:
                investor_data[key] = []
            investor_data[key].append(value)
        
        for key, value in better_data.items():
            if key not in better_deal_data:
                better_deal_data[key] = []
            better_deal_data[key].append(value)
        
        for key, value in worse_data.items():
            if key not in worse_deal_data:
                worse_deal_data[key] = []
            worse_deal_data[key].append(value)
    
    # Convert to tensors
    def to_tensors(data_dict: dict[str, list]) -> dict[str, torch.Tensor]:
        tensors = {}
        for key, values in data_dict.items():
            if key in ["min_investment", "max_investment", "experience_years", 
                      "portfolio_size", "deal_size", "revenue_multiple", 
                      "growth_rate", "profitability", "team_experience", "market_size"]:
                tensors[key] = torch.tensor(values, dtype=torch.float32)
            else:
                tensors[key] = torch.tensor(values, dtype=torch.long)
        return tensors
    
    return (
        to_tensors(investor_data),
        to_tensors(better_deal_data),
        to_tensors(worse_deal_data),
    )
