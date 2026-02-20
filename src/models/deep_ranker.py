"""
Deep Ranking Model with Pairwise RankNet Loss.

This model is used for re-ranking candidates retrieved by the Two-Tower model.
It concatenates investor and deal features and learns a scoring function
optimized for pairwise ranking.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from src.config import FeatureDims, DeepRankingConfig, get_settings


class PairwiseDeepRankingModel(pl.LightningModule):
    """
    Deep Ranking Model with Pairwise RankNet Loss.
    
    Unlike the Two-Tower model which uses separate encoders, this model
    concatenates investor and deal features and learns a joint scoring
    function. It's trained with pairwise loss (RankNet) to directly
    optimize for ranking quality.
    
    Architecture:
    1. Investor embeddings (ID + categorical + numerical)
    2. Deal embeddings (ID + categorical + numerical)
    3. Concatenate investor and deal features
    4. Deep MLP with BatchNorm and Dropout
    5. Single output score
    
    Args:
        n_investors: Total number of investors
        n_deals: Total number of deals
        feature_dims: Feature dimension configuration
        config: Model configuration (optional)
        learning_rate: Override learning rate (optional)
    """
    
    def __init__(
        self,
        n_investors: int,
        n_deals: int,
        feature_dims: FeatureDims | dict,
        config: Optional[DeepRankingConfig] = None,
        learning_rate: Optional[float] = None,
    ):
        super().__init__()
        
        # Handle dict input for feature_dims
        if isinstance(feature_dims, dict):
            feature_dims = FeatureDims(**feature_dims)
        
        if config is None:
            config = get_settings().deep_ranking
        
        self.save_hyperparameters(ignore=["config"])
        
        self.n_investors = n_investors
        self.n_deals = n_deals
        self.feature_dims = feature_dims
        self.config = config
        self.learning_rate = learning_rate or get_settings().training.learning_rate
        
        # Investor embeddings
        self.investor_embedding = nn.Embedding(n_investors, 64)
        self.investor_type_embedding = nn.Embedding(feature_dims.n_investor_types, 16)
        self.investor_region_embedding = nn.Embedding(feature_dims.n_regions, 16)
        self.investor_risk_embedding = nn.Embedding(feature_dims.n_risk_profiles, 8)
        
        # Deal embeddings
        self.deal_embedding = nn.Embedding(n_deals, 64)
        self.deal_sector_embedding = nn.Embedding(feature_dims.n_sectors, 16)
        self.deal_stage_embedding = nn.Embedding(feature_dims.n_stages, 16)
        self.deal_region_embedding = nn.Embedding(feature_dims.n_regions, 16)
        
        # Calculate input dimension
        # Investor: 64 + 16 + 16 + 8 + 4 = 108
        # Deal: 64 + 16 + 16 + 16 + 6 = 118
        # Total: 226
        investor_dim = 64 + 16 + 16 + 8 + 4
        deal_dim = 64 + 16 + 16 + 16 + 6
        input_dim = investor_dim + deal_dim
        
        # Deep MLP
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer - single score
        layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embedding weights."""
        for embedding in [
            self.investor_embedding, self.investor_type_embedding,
            self.investor_region_embedding, self.investor_risk_embedding,
            self.deal_embedding, self.deal_sector_embedding,
            self.deal_stage_embedding, self.deal_region_embedding,
        ]:
            nn.init.xavier_uniform_(embedding.weight)
    
    def forward(
        self, 
        investor_data: dict[str, torch.Tensor], 
        deal_data: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute ranking scores for investor-deal pairs.
        
        Args:
            investor_data: Dict of investor features
            deal_data: Dict of deal features
            
        Returns:
            Tensor of shape (batch_size, 1) - ranking scores
        """
        # Investor embeddings
        inv_id_emb = self.investor_embedding(investor_data["id"])
        inv_type_emb = self.investor_type_embedding(investor_data["type"])
        inv_region_emb = self.investor_region_embedding(investor_data["region"])
        inv_risk_emb = self.investor_risk_embedding(investor_data["risk"])
        
        inv_numerical = torch.stack([
            investor_data["min_investment"],
            investor_data["max_investment"],
            investor_data["experience_years"],
            investor_data["portfolio_size"],
        ], dim=-1)
        
        # Deal embeddings
        deal_id_emb = self.deal_embedding(deal_data["id"])
        deal_sector_emb = self.deal_sector_embedding(deal_data["sector"])
        deal_stage_emb = self.deal_stage_embedding(deal_data["stage"])
        deal_region_emb = self.deal_region_embedding(deal_data["region"])
        
        deal_numerical = torch.stack([
            deal_data["deal_size"],
            deal_data["revenue_multiple"],
            deal_data["growth_rate"],
            deal_data["profitability"],
            deal_data["team_experience"],
            deal_data["market_size"],
        ], dim=-1)
        
        # Concatenate investor features
        investor_features = torch.cat([
            inv_id_emb, inv_type_emb, inv_region_emb, inv_risk_emb, inv_numerical
        ], dim=-1)
        
        # Concatenate deal features
        deal_features = torch.cat([
            deal_id_emb, deal_sector_emb, deal_stage_emb, deal_region_emb, deal_numerical
        ], dim=-1)
        
        # Combine investor and deal features
        combined = torch.cat([investor_features, deal_features], dim=-1)
        
        # Pass through MLP
        score = self.mlp(combined)
        
        return score
    
    def training_step(
        self, 
        batch: tuple[dict, dict, dict], 
        batch_idx: int
    ) -> torch.Tensor:
        """
        Training step with pairwise RankNet loss.
        
        The model is trained to score better_deal higher than worse_deal
        for each investor.
        """
        investor_data, better_deal_data, worse_deal_data = batch
        
        # Get scores for both deals
        better_scores = self(investor_data, better_deal_data).squeeze(-1)
        worse_scores = self(investor_data, worse_deal_data).squeeze(-1)
        
        # RankNet loss: BCE on score difference
        # P(better > worse) = sigmoid(better_score - worse_score)
        # Target = 1 (better should always be ranked higher)
        loss = F.binary_cross_entropy_with_logits(
            better_scores - worse_scores,
            torch.ones_like(better_scores)
        )
        
        # Track accuracy (% of pairs correctly ranked)
        with torch.no_grad():
            correct = (better_scores > worse_scores).float().mean()
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_accuracy", correct, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(
        self, 
        batch: tuple[dict, dict, dict], 
        batch_idx: int
    ) -> torch.Tensor:
        """Validation step."""
        investor_data, better_deal_data, worse_deal_data = batch
        
        better_scores = self(investor_data, better_deal_data).squeeze(-1)
        worse_scores = self(investor_data, worse_deal_data).squeeze(-1)
        
        loss = F.binary_cross_entropy_with_logits(
            better_scores - worse_scores,
            torch.ones_like(better_scores)
        )
        
        correct = (better_scores > worse_scores).float().mean()
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_accuracy", correct, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(
        self, 
        batch: tuple[dict, dict, dict], 
        batch_idx: int
    ) -> torch.Tensor:
        """Test step."""
        investor_data, better_deal_data, worse_deal_data = batch
        
        better_scores = self(investor_data, better_deal_data).squeeze(-1)
        worse_scores = self(investor_data, worse_deal_data).squeeze(-1)
        
        loss = F.binary_cross_entropy_with_logits(
            better_scores - worse_scores,
            torch.ones_like(better_scores)
        )
        
        correct = (better_scores > worse_scores).float().mean()
        
        self.log("test_loss", loss)
        self.log("test_accuracy", correct)
        
        return loss
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer."""
        return torch.optim.Adam(
            self.parameters(), 
            lr=self.learning_rate,
            weight_decay=get_settings().training.weight_decay,
        )
    
    def score_deals(
        self,
        investor_data: dict[str, torch.Tensor],
        deal_data: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Score a batch of deals for an investor (for inference).
        
        Args:
            investor_data: Investor features (can be single investor repeated)
            deal_data: Deal features for multiple deals
            
        Returns:
            Tensor of shape (n_deals,) - scores for each deal
        """
        self.eval()
        with torch.no_grad():
            scores = self(investor_data, deal_data)
            return scores.squeeze(-1)
