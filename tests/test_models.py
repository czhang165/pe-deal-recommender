"""Basic tests for the recommender system."""

import pytest
import torch
import numpy as np

from src.config import Settings, FeatureDims, TwoTowerConfig, DeepRankingConfig
from src.models import TwoTowerModel, PairwiseDeepRankingModel
from src.models.towers import InvestorTower, DealTower


@pytest.fixture
def feature_dims():
    """Sample feature dimensions."""
    return FeatureDims(
        n_investor_types=3,
        n_regions=4,
        n_risk_profiles=3,
        n_sectors=5,
        n_stages=5,
    )


@pytest.fixture
def two_tower_config():
    """Two-tower model config."""
    return TwoTowerConfig(
        embedding_dim=32,
        tower_hidden_dims=[64, 32],
        temperature=0.1,
    )


@pytest.fixture
def deep_ranking_config():
    """Deep ranking model config."""
    return DeepRankingConfig(
        hidden_dims=[64, 32],
        dropout_rate=0.2,
    )


@pytest.fixture
def sample_investor_batch():
    """Sample investor batch."""
    batch_size = 4
    return {
        "id": torch.tensor([0, 1, 2, 3]),
        "type": torch.tensor([0, 1, 2, 0]),
        "region": torch.tensor([0, 1, 2, 3]),
        "risk": torch.tensor([0, 1, 2, 0]),
        "min_investment": torch.tensor([1.0, 2.0, 3.0, 4.0]),
        "max_investment": torch.tensor([10.0, 20.0, 30.0, 40.0]),
        "experience_years": torch.tensor([5.0, 10.0, 15.0, 20.0]),
        "portfolio_size": torch.tensor([10.0, 20.0, 30.0, 40.0]),
    }


@pytest.fixture
def sample_deal_batch():
    """Sample deal batch."""
    batch_size = 4
    return {
        "id": torch.tensor([0, 1, 2, 3]),
        "sector": torch.tensor([0, 1, 2, 3]),
        "stage": torch.tensor([0, 1, 2, 3]),
        "region": torch.tensor([0, 1, 2, 3]),
        "deal_size": torch.tensor([10.0, 20.0, 30.0, 40.0]),
        "revenue_multiple": torch.tensor([2.0, 3.0, 4.0, 5.0]),
        "growth_rate": torch.tensor([0.1, 0.2, 0.3, 0.4]),
        "profitability": torch.tensor([0.05, 0.1, 0.15, 0.2]),
        "team_experience": torch.tensor([5.0, 10.0, 15.0, 20.0]),
        "market_size": torch.tensor([100.0, 200.0, 300.0, 400.0]),
    }


class TestInvestorTower:
    """Tests for InvestorTower."""
    
    def test_forward_shape(self, feature_dims, two_tower_config, sample_investor_batch):
        tower = InvestorTower(
            n_investors=10,
            feature_dims=feature_dims,
            config=two_tower_config,
        )
        
        output = tower(sample_investor_batch)
        
        assert output.shape == (4, 32)  # (batch_size, output_dim)
    
    def test_l2_normalized(self, feature_dims, two_tower_config, sample_investor_batch):
        tower = InvestorTower(
            n_investors=10,
            feature_dims=feature_dims,
            config=two_tower_config,
        )
        
        output = tower(sample_investor_batch)
        
        # Check L2 normalization
        norms = torch.norm(output, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


class TestDealTower:
    """Tests for DealTower."""
    
    def test_forward_shape(self, feature_dims, two_tower_config, sample_deal_batch):
        tower = DealTower(
            n_deals=10,
            feature_dims=feature_dims,
            config=two_tower_config,
        )
        
        output = tower(sample_deal_batch)
        
        assert output.shape == (4, 32)


class TestTwoTowerModel:
    """Tests for TwoTowerModel."""
    
    def test_forward(
        self, feature_dims, two_tower_config, 
        sample_investor_batch, sample_deal_batch
    ):
        model = TwoTowerModel(
            n_investors=10,
            n_deals=10,
            feature_dims=feature_dims,
            config=two_tower_config,
        )
        
        similarities = model(sample_investor_batch, sample_deal_batch)
        
        assert similarities.shape == (4,)
    
    def test_training_step(
        self, feature_dims, two_tower_config,
        sample_investor_batch, sample_deal_batch
    ):
        model = TwoTowerModel(
            n_investors=10,
            n_deals=10,
            feature_dims=feature_dims,
            config=two_tower_config,
        )
        
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        batch = (sample_investor_batch, sample_deal_batch, labels)
        
        # Mock trainer
        model.trainer = type('obj', (object,), {'logged_metrics': {}})()
        
        loss = model.training_step(batch, 0)
        
        assert loss.shape == ()
        assert loss.item() > 0


class TestPairwiseDeepRankingModel:
    """Tests for PairwiseDeepRankingModel."""
    
    def test_forward(
        self, feature_dims, deep_ranking_config,
        sample_investor_batch, sample_deal_batch
    ):
        model = PairwiseDeepRankingModel(
            n_investors=10,
            n_deals=10,
            feature_dims=feature_dims,
            config=deep_ranking_config,
        )
        
        scores = model(sample_investor_batch, sample_deal_batch)
        
        assert scores.shape == (4, 1)
    
    def test_training_step(
        self, feature_dims, deep_ranking_config,
        sample_investor_batch, sample_deal_batch
    ):
        model = PairwiseDeepRankingModel(
            n_investors=10,
            n_deals=10,
            feature_dims=feature_dims,
            config=deep_ranking_config,
        )
        
        # Create worse deal batch (different IDs)
        worse_deal_batch = sample_deal_batch.copy()
        worse_deal_batch["id"] = torch.tensor([4, 5, 6, 7])
        
        batch = (sample_investor_batch, sample_deal_batch, worse_deal_batch)
        
        # Mock trainer
        model.trainer = type('obj', (object,), {'logged_metrics': {}})()
        
        loss = model.training_step(batch, 0)
        
        assert loss.shape == ()
        assert loss.item() > 0


class TestSettings:
    """Tests for Settings configuration."""
    
    def test_default_settings(self):
        settings = Settings()
        
        assert settings.environment.value == "local"
        assert settings.storage_backend.value == "local"
        assert settings.data.n_investors == 125
        assert settings.data.n_deals == 1000
    
    def test_feature_dims(self, feature_dims):
        assert feature_dims.n_investor_types == 3
        assert feature_dims.n_regions == 4
        
        dims_dict = feature_dims.to_dict()
        assert isinstance(dims_dict, dict)
        assert "n_investor_types" in dims_dict
