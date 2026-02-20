"""
Tower modules for the Two-Tower architecture.

Contains:
- InvestorTower: Encodes investor features to embeddings
- DealTower: Encodes deal features to embeddings
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import FeatureDims, TwoTowerConfig


class InvestorTower(nn.Module):
    """
    Neural network tower for encoding investor features.
    
    Architecture:
    1. ID embedding for collaborative filtering signal
    2. Categorical embeddings (type, region, risk profile)
    3. Numerical features (investment range, experience, portfolio size)
    4. MLP layers with BatchNorm and Dropout
    5. Optional L2 normalization for similarity computation
    
    Args:
        n_investors: Total number of investors
        feature_dims: Feature dimension configuration
        config: Two-tower model configuration
    """
    
    def __init__(
        self,
        n_investors: int,
        feature_dims: FeatureDims,
        config: TwoTowerConfig,
    ):
        super().__init__()
        self.config = config
        
        # ID embedding
        self.investor_embedding = nn.Embedding(
            n_investors, config.embedding_dim
        )
        
        # Categorical embeddings
        self.type_embedding = nn.Embedding(feature_dims.n_investor_types, 16)
        self.region_embedding = nn.Embedding(feature_dims.n_regions, 16)
        self.risk_embedding = nn.Embedding(feature_dims.n_risk_profiles, 8)
        
        # Calculate total input dimension
        # ID (embedding_dim) + type (16) + region (16) + risk (8) + numerical (4)
        input_dim = config.embedding_dim + 16 + 16 + 8 + 4
        
        # MLP layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in config.tower_hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(config.dropout_rate),
            ])
            prev_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        self.output_dim = config.tower_hidden_dims[-1]
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embedding weights with Xavier uniform."""
        nn.init.xavier_uniform_(self.investor_embedding.weight)
        nn.init.xavier_uniform_(self.type_embedding.weight)
        nn.init.xavier_uniform_(self.region_embedding.weight)
        nn.init.xavier_uniform_(self.risk_embedding.weight)
    
    def forward(self, investor_data: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the investor tower.
        
        Args:
            investor_data: Dict containing:
                - id: (batch_size,) investor IDs
                - type: (batch_size,) investor type indices
                - region: (batch_size,) region indices
                - risk: (batch_size,) risk profile indices
                - min_investment: (batch_size,) min investment values
                - max_investment: (batch_size,) max investment values
                - experience_years: (batch_size,) years of experience
                - portfolio_size: (batch_size,) portfolio size
                
        Returns:
            Tensor of shape (batch_size, output_dim) - investor embeddings
        """
        # Get embeddings
        id_emb = self.investor_embedding(investor_data["id"])
        type_emb = self.type_embedding(investor_data["type"])
        region_emb = self.region_embedding(investor_data["region"])
        risk_emb = self.risk_embedding(investor_data["risk"])
        
        # Stack numerical features
        numerical = torch.stack([
            investor_data["min_investment"],
            investor_data["max_investment"],
            investor_data["experience_years"],
            investor_data["portfolio_size"],
        ], dim=-1)
        
        # Concatenate all features
        x = torch.cat([id_emb, type_emb, region_emb, risk_emb, numerical], dim=-1)
        
        # Pass through MLP
        embedding = self.mlp(x)
        
        # Optional L2 normalization
        if self.config.l2_normalize:
            embedding = F.normalize(embedding, p=2, dim=-1)
        
        return embedding


class DealTower(nn.Module):
    """
    Neural network tower for encoding deal features.
    
    Architecture:
    1. ID embedding for collaborative filtering signal
    2. Categorical embeddings (sector, stage, region)
    3. Numerical features (deal size, revenue multiple, growth, etc.)
    4. MLP layers with BatchNorm and Dropout
    5. Optional L2 normalization for similarity computation
    
    Args:
        n_deals: Total number of deals
        feature_dims: Feature dimension configuration
        config: Two-tower model configuration
    """
    
    def __init__(
        self,
        n_deals: int,
        feature_dims: FeatureDims,
        config: TwoTowerConfig,
    ):
        super().__init__()
        self.config = config
        
        # ID embedding
        self.deal_embedding = nn.Embedding(n_deals, config.embedding_dim)
        
        # Categorical embeddings
        self.sector_embedding = nn.Embedding(feature_dims.n_sectors, 16)
        self.stage_embedding = nn.Embedding(feature_dims.n_stages, 16)
        self.region_embedding = nn.Embedding(feature_dims.n_regions, 16)
        
        # Calculate total input dimension
        # ID (embedding_dim) + sector (16) + stage (16) + region (16) + numerical (6)
        input_dim = config.embedding_dim + 16 + 16 + 16 + 6
        
        # MLP layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in config.tower_hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(config.dropout_rate),
            ])
            prev_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        self.output_dim = config.tower_hidden_dims[-1]
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embedding weights with Xavier uniform."""
        nn.init.xavier_uniform_(self.deal_embedding.weight)
        nn.init.xavier_uniform_(self.sector_embedding.weight)
        nn.init.xavier_uniform_(self.stage_embedding.weight)
        nn.init.xavier_uniform_(self.region_embedding.weight)
    
    def forward(self, deal_data: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the deal tower.
        
        Args:
            deal_data: Dict containing:
                - id: (batch_size,) deal IDs
                - sector: (batch_size,) sector indices
                - stage: (batch_size,) stage indices
                - region: (batch_size,) region indices
                - deal_size: (batch_size,) deal size values
                - revenue_multiple: (batch_size,) revenue multiple values
                - growth_rate: (batch_size,) growth rate values
                - profitability: (batch_size,) profitability values
                - team_experience: (batch_size,) team experience values
                - market_size: (batch_size,) market size values
                
        Returns:
            Tensor of shape (batch_size, output_dim) - deal embeddings
        """
        # Get embeddings
        id_emb = self.deal_embedding(deal_data["id"])
        sector_emb = self.sector_embedding(deal_data["sector"])
        stage_emb = self.stage_embedding(deal_data["stage"])
        region_emb = self.region_embedding(deal_data["region"])
        
        # Stack numerical features
        numerical = torch.stack([
            deal_data["deal_size"],
            deal_data["revenue_multiple"],
            deal_data["growth_rate"],
            deal_data["profitability"],
            deal_data["team_experience"],
            deal_data["market_size"],
        ], dim=-1)
        
        # Concatenate all features
        x = torch.cat([id_emb, sector_emb, stage_emb, region_emb, numerical], dim=-1)
        
        # Pass through MLP
        embedding = self.mlp(x)
        
        # Optional L2 normalization
        if self.config.l2_normalize:
            embedding = F.normalize(embedding, p=2, dim=-1)
        
        return embedding
