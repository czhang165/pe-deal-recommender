"""
Feature encoding utilities for investor and deal data.

Handles encoding of categorical features and normalization of numerical features.
Serializable for use in production inference.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import FeatureDims


@dataclass
class FeatureEncoder:
    """
    Encodes categorical and numerical features for investors and deals.
    
    This encoder:
    - Maps categorical features to integer indices
    - Tracks feature dimensions for embedding layers
    - Provides serialization for deployment
    
    Example:
        encoder = FeatureEncoder()
        encoder.fit(investor_df, deal_df)
        
        # Get encoded features
        inv_features = encoder.encode_investor_features(investor_df)
        deal_features = encoder.encode_deal_features(deal_df)
        
        # Save for deployment
        encoder.save("models/feature_encoder.pkl")
    """
    
    # Categorical encoders: {feature_name: {value: index}}
    encoders: dict[str, dict[Any, int]] = field(default_factory=dict)
    
    # Feature statistics for numerical normalization
    numerical_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    
    # Derived feature dimensions
    feature_dims: FeatureDims = field(default_factory=FeatureDims)
    
    # Flag to check if fitted
    _is_fitted: bool = field(default=False, repr=False)
    
    def fit(self, investor_df: pd.DataFrame, deal_df: pd.DataFrame) -> "FeatureEncoder":
        """
        Fit the encoder on investor and deal dataframes.
        
        Args:
            investor_df: DataFrame with investor features
            deal_df: DataFrame with deal features
            
        Returns:
            Self for method chaining
        """
        # Encode investor categorical features
        self.encoders["investor_type"] = self._create_encoder(investor_df["type"])
        self.encoders["investor_region"] = self._create_encoder(investor_df["preferred_region"])
        self.encoders["investor_risk"] = self._create_encoder(investor_df["risk_profile"])
        
        # Encode deal categorical features
        self.encoders["deal_sector"] = self._create_encoder(deal_df["sector"])
        self.encoders["deal_stage"] = self._create_encoder(deal_df["stage"])
        self.encoders["deal_region"] = self._create_encoder(deal_df["region"])
        
        # Compute numerical statistics for normalization
        self._compute_numerical_stats(investor_df, deal_df)
        
        # Set feature dimensions
        self.feature_dims = FeatureDims(
            n_investor_types=len(self.encoders["investor_type"]),
            n_regions=max(
                len(self.encoders["investor_region"]),
                len(self.encoders["deal_region"])
            ),
            n_risk_profiles=len(self.encoders["investor_risk"]),
            n_sectors=len(self.encoders["deal_sector"]),
            n_stages=len(self.encoders["deal_stage"]),
        )
        
        self._is_fitted = True
        return self
    
    def _create_encoder(self, series: pd.Series) -> dict[Any, int]:
        """Create a mapping from unique values to indices."""
        unique_values = sorted(series.unique())
        return {v: i for i, v in enumerate(unique_values)}
    
    def _compute_numerical_stats(
        self, 
        investor_df: pd.DataFrame, 
        deal_df: pd.DataFrame
    ) -> None:
        """Compute mean and std for numerical features."""
        # Investor numerical features
        investor_numerical = ["min_investment", "max_investment", "experience_years", "portfolio_size"]
        for col in investor_numerical:
            if col in investor_df.columns:
                self.numerical_stats[f"investor_{col}"] = {
                    "mean": float(investor_df[col].mean()),
                    "std": float(investor_df[col].std()) or 1.0,
                }
        
        # Deal numerical features
        deal_numerical = ["deal_size", "revenue_multiple", "growth_rate", 
                         "profitability", "team_experience", "market_size"]
        for col in deal_numerical:
            if col in deal_df.columns:
                self.numerical_stats[f"deal_{col}"] = {
                    "mean": float(deal_df[col].mean()),
                    "std": float(deal_df[col].std()) or 1.0,
                }
    
    def encode_investor_features(
        self, 
        investor_df: pd.DataFrame,
        normalize_numerical: bool = False
    ) -> dict[str, np.ndarray]:
        """
        Encode investor features to numpy arrays.
        
        Args:
            investor_df: DataFrame with investor features (indexed by investorId)
            normalize_numerical: Whether to z-score normalize numerical features
            
        Returns:
            Dictionary mapping feature names to numpy arrays
        """
        self._check_fitted()
        
        # Ensure we have proper indexing
        if "investorId" in investor_df.columns:
            investor_df = investor_df.set_index("investorId")
        
        features = {
            "type": investor_df["type"].map(self.encoders["investor_type"]).values.astype(np.int64),
            "region": investor_df["preferred_region"].map(self.encoders["investor_region"]).values.astype(np.int64),
            "risk": investor_df["risk_profile"].map(self.encoders["investor_risk"]).values.astype(np.int64),
            "min_investment": investor_df["min_investment"].values.astype(np.float32),
            "max_investment": investor_df["max_investment"].values.astype(np.float32),
            "experience_years": investor_df["experience_years"].values.astype(np.float32),
            "portfolio_size": investor_df["portfolio_size"].values.astype(np.float32),
        }
        
        if normalize_numerical:
            for col in ["min_investment", "max_investment", "experience_years", "portfolio_size"]:
                stats = self.numerical_stats.get(f"investor_{col}")
                if stats:
                    features[col] = (features[col] - stats["mean"]) / stats["std"]
        
        return features
    
    def encode_deal_features(
        self, 
        deal_df: pd.DataFrame,
        normalize_numerical: bool = False
    ) -> dict[str, np.ndarray]:
        """
        Encode deal features to numpy arrays.
        
        Args:
            deal_df: DataFrame with deal features (indexed by dealId)
            normalize_numerical: Whether to z-score normalize numerical features
            
        Returns:
            Dictionary mapping feature names to numpy arrays
        """
        self._check_fitted()
        
        # Ensure we have proper indexing
        if "dealId" in deal_df.columns:
            deal_df = deal_df.set_index("dealId")
        
        features = {
            "sector": deal_df["sector"].map(self.encoders["deal_sector"]).values.astype(np.int64),
            "stage": deal_df["stage"].map(self.encoders["deal_stage"]).values.astype(np.int64),
            "region": deal_df["region"].map(self.encoders["deal_region"]).values.astype(np.int64),
            "deal_size": deal_df["deal_size"].values.astype(np.float32),
            "revenue_multiple": deal_df["revenue_multiple"].values.astype(np.float32),
            "growth_rate": deal_df["growth_rate"].values.astype(np.float32),
            "profitability": deal_df["profitability"].values.astype(np.float32),
            "team_experience": deal_df["team_experience"].values.astype(np.float32),
            "market_size": deal_df["market_size"].values.astype(np.float32),
        }
        
        if normalize_numerical:
            for col in ["deal_size", "revenue_multiple", "growth_rate", 
                       "profitability", "team_experience", "market_size"]:
                stats = self.numerical_stats.get(f"deal_{col}")
                if stats:
                    features[col] = (features[col] - stats["mean"]) / stats["std"]
        
        return features
    
    def _check_fitted(self) -> None:
        """Check if encoder has been fitted."""
        if not self._is_fitted:
            raise RuntimeError("FeatureEncoder has not been fitted. Call fit() first.")
    
    def save(self, path: str | Path) -> None:
        """Save encoder to pickle file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str | Path) -> "FeatureEncoder":
        """Load encoder from pickle file."""
        with open(path, "rb") as f:
            return pickle.load(f)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert encoder to dictionary for JSON serialization."""
        return {
            "encoders": self.encoders,
            "numerical_stats": self.numerical_stats,
            "feature_dims": self.feature_dims.model_dump(),
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeatureEncoder":
        """Create encoder from dictionary."""
        encoder = cls()
        encoder.encoders = data["encoders"]
        encoder.numerical_stats = data["numerical_stats"]
        encoder.feature_dims = FeatureDims(**data["feature_dims"])
        encoder._is_fitted = True
        return encoder
    
    def __repr__(self) -> str:
        if not self._is_fitted:
            return "FeatureEncoder(not fitted)"
        return f"FeatureEncoder(feature_dims={self.feature_dims})"
