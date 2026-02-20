"""
Two-Tower Model for candidate retrieval.

The Two-Tower architecture encodes investors and deals separately,
then computes similarity via dot product. This allows efficient
retrieval using approximate nearest neighbor search.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from src.config import FeatureDims, TwoTowerConfig, get_settings
from src.models.towers import InvestorTower, DealTower


class TwoTowerModel(pl.LightningModule):
    """
    Two-Tower Neural Network for retrieval.
    
    This model learns separate embeddings for investors and deals,
    trained to maximize similarity for positive pairs and minimize
    for negative pairs using binary cross-entropy loss.
    
    Key features:
    - Separate towers allow pre-computing deal embeddings for fast retrieval
    - Temperature-scaled dot product similarity
    - L2 normalized embeddings for cosine similarity
    
    Args:
        n_investors: Total number of investors
        n_deals: Total number of deals
        feature_dims: Feature dimension configuration
        config: Model configuration (optional, uses settings if not provided)
        learning_rate: Override learning rate (optional)
    """
    
    def __init__(
        self,
        n_investors: int,
        n_deals: int,
        feature_dims: FeatureDims | dict,
        config: Optional[TwoTowerConfig] = None,
        learning_rate: Optional[float] = None,
    ):
        super().__init__()
        
        # Handle dict input for feature_dims (from checkpoint loading)
        if isinstance(feature_dims, dict):
            feature_dims = FeatureDims(**feature_dims)
        
        # Use provided config or get from settings
        if config is None:
            config = get_settings().two_tower
        
        self.save_hyperparameters(ignore=["config"])
        
        self.n_investors = n_investors
        self.n_deals = n_deals
        self.feature_dims = feature_dims
        self.config = config
        self.learning_rate = learning_rate or get_settings().training.learning_rate
        
        # Build towers
        self.investor_tower = InvestorTower(n_investors, feature_dims, config)
        self.deal_tower = DealTower(n_deals, feature_dims, config)
        
        self.temperature = config.temperature
        
        # For tracking losses (used by callbacks)
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
    
    def forward(
        self, 
        investor_data: dict[str, torch.Tensor], 
        deal_data: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute similarity scores between investors and deals.
        
        Args:
            investor_data: Dict of investor features
            deal_data: Dict of deal features
            
        Returns:
            Tensor of shape (batch_size,) - similarity scores
        """
        investor_emb = self.investor_tower(investor_data)
        deal_emb = self.deal_tower(deal_data)
        
        # Dot product similarity with temperature scaling
        similarities = torch.sum(investor_emb * deal_emb, dim=-1) / self.temperature
        
        return similarities
    
    def encode_investor(
        self, investor_data: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Encode investor to embedding (for inference)."""
        return self.investor_tower(investor_data)
    
    def encode_deal(self, deal_data: dict[str, torch.Tensor]) -> torch.Tensor:
        """Encode deal to embedding (for inference)."""
        return self.deal_tower(deal_data)
    
    def training_step(
        self, 
        batch: tuple[dict, dict, torch.Tensor], 
        batch_idx: int
    ) -> torch.Tensor:
        """Training step with binary cross-entropy loss."""
        investor_data, deal_data, labels = batch
        
        similarities = self(investor_data, deal_data)
        loss = F.binary_cross_entropy_with_logits(similarities, labels)
        
        # Compute accuracy for monitoring
        with torch.no_grad():
            preds = (torch.sigmoid(similarities) > 0.5).float()
            acc = (preds == labels).float().mean()
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(
        self, 
        batch: tuple[dict, dict, torch.Tensor], 
        batch_idx: int
    ) -> torch.Tensor:
        """Validation step."""
        investor_data, deal_data, labels = batch
        
        similarities = self(investor_data, deal_data)
        loss = F.binary_cross_entropy_with_logits(similarities, labels)
        
        # Compute accuracy
        preds = (torch.sigmoid(similarities) > 0.5).float()
        acc = (preds == labels).float().mean()
        
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(
        self, 
        batch: tuple[dict, dict, torch.Tensor], 
        batch_idx: int
    ) -> torch.Tensor:
        """Test step."""
        investor_data, deal_data, labels = batch
        
        similarities = self(investor_data, deal_data)
        loss = F.binary_cross_entropy_with_logits(similarities, labels)
        
        preds = (torch.sigmoid(similarities) > 0.5).float()
        acc = (preds == labels).float().mean()
        
        self.log("test_loss", loss)
        self.log("test_acc", acc)
        
        return loss
    
    def on_train_epoch_end(self) -> None:
        """Record training loss at epoch end."""
        if "train_loss" in self.trainer.logged_metrics:
            loss = self.trainer.logged_metrics["train_loss"]
            if hasattr(loss, "item"):
                self.train_losses.append(loss.item())
            else:
                self.train_losses.append(float(loss))
    
    def on_validation_epoch_end(self) -> None:
        """Record validation loss at epoch end."""
        if "val_loss" in self.trainer.logged_metrics:
            loss = self.trainer.logged_metrics["val_loss"]
            if hasattr(loss, "item"):
                self.val_losses.append(loss.item())
            else:
                self.val_losses.append(float(loss))
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure optimizer."""
        return torch.optim.Adam(
            self.parameters(), 
            lr=self.learning_rate,
            weight_decay=get_settings().training.weight_decay,
        )
    
    def get_deal_embeddings(
        self, 
        deal_data: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Get embeddings for all deals (for building FAISS index).
        
        Args:
            deal_data: Dict of deal features for all deals
            
        Returns:
            Tensor of shape (n_deals, embedding_dim)
        """
        self.eval()
        with torch.no_grad():
            return self.deal_tower(deal_data)
