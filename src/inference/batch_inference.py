"""
Inference module for the recommender system.

Provides:
- FAISS-based vector index for fast retrieval
- Batch inference utilities
- Two-stage inference pipeline
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from src.config import Settings, get_settings
from src.data import FeatureEncoder
from src.models import TwoTowerModel, PairwiseDeepRankingModel

# FAISS is optional but recommended
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


def check_faiss_available() -> None:
    """Raise error if FAISS is not installed."""
    if not FAISS_AVAILABLE:
        raise ImportError(
            "FAISS is not installed. Install with: pip install faiss-cpu"
        )


@dataclass
class RecommendationResult:
    """Container for recommendation results."""
    
    investor_id: int
    recommended_deal_ids: list[int]
    scores: list[float]
    retrieval_candidates: int = 0
    
    def to_dict(self) -> dict:
        return {
            "investor_id": self.investor_id,
            "recommendations": [
                {"deal_id": did, "score": score}
                for did, score in zip(self.recommended_deal_ids, self.scores)
            ],
            "retrieval_candidates": self.retrieval_candidates,
        }


class FAISSIndex:
    """
    FAISS-based vector index for fast approximate nearest neighbor search.
    
    Supports:
    - FlatIP: Exact inner product search (default)
    - IVFFlat: Inverted file index for larger datasets
    """
    
    def __init__(
        self,
        embedding_dim: int,
        index_type: str = "FlatIP",
        nlist: int = 100,  # For IVF indices
    ):
        check_faiss_available()
        
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        
        if index_type == "FlatIP":
            # Exact inner product search
            self.index = faiss.IndexFlatIP(embedding_dim)
        elif index_type == "IVFFlat":
            # Inverted file index for larger datasets
            quantizer = faiss.IndexFlatIP(embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        self.deal_ids: Optional[np.ndarray] = None
        self.is_trained = False
    
    def build(self, embeddings: np.ndarray, deal_ids: np.ndarray) -> None:
        """
        Build the index from deal embeddings.
        
        Args:
            embeddings: (n_deals, embedding_dim) array of embeddings
            deal_ids: (n_deals,) array of deal IDs
        """
        # Ensure embeddings are float32 and contiguous
        embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
        
        # Train if needed (for IVF indices)
        if hasattr(self.index, "train") and not self.index.is_trained:
            self.index.train(embeddings)
        
        # Add embeddings
        self.index.add(embeddings)
        self.deal_ids = deal_ids
        self.is_trained = True
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        k: int = 100,
        nprobe: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Search for nearest neighbors.
        
        Args:
            query_embedding: (1, embedding_dim) or (embedding_dim,) query vector
            k: Number of neighbors to return
            nprobe: Number of clusters to search (for IVF indices)
            
        Returns:
            Tuple of (deal_ids, scores)
        """
        if not self.is_trained:
            raise RuntimeError("Index not built. Call build() first.")
        
        # Ensure correct shape
        query = np.ascontiguousarray(query_embedding.astype(np.float32))
        if query.ndim == 1:
            query = query.reshape(1, -1)
        
        # Set nprobe for IVF indices
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = nprobe
        
        # Search
        scores, indices = self.index.search(query, k)
        
        # Map indices to deal IDs
        deal_ids = self.deal_ids[indices[0]]
        
        return deal_ids, scores[0]
    
    def save(self, path: str | Path) -> None:
        """Save index to file."""
        check_faiss_available()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))
        
        # Save deal IDs separately
        np.save(str(path) + ".ids.npy", self.deal_ids)
    
    @classmethod
    def load(cls, path: str | Path, embedding_dim: int) -> "FAISSIndex":
        """Load index from file."""
        check_faiss_available()
        path = Path(path)
        
        instance = cls.__new__(cls)
        instance.embedding_dim = embedding_dim
        instance.index = faiss.read_index(str(path))
        instance.deal_ids = np.load(str(path) + ".ids.npy")
        instance.is_trained = True
        
        return instance


class RecommenderInference:
    """
    Two-stage inference pipeline.
    
    Uses:
    1. Two-Tower model + FAISS for fast candidate retrieval
    2. Deep Ranking model for re-ranking
    """
    
    def __init__(
        self,
        two_tower_model: TwoTowerModel,
        deep_ranking_model: PairwiseDeepRankingModel,
        encoder: FeatureEncoder,
        investor_features: dict[str, np.ndarray],
        deal_features: dict[str, np.ndarray],
        faiss_index: Optional[FAISSIndex] = None,
        settings: Optional[Settings] = None,
    ):
        self.two_tower_model = two_tower_model
        self.deep_ranking_model = deep_ranking_model
        self.encoder = encoder
        self.investor_features = investor_features
        self.deal_features = deal_features
        self.faiss_index = faiss_index
        self.settings = settings or get_settings()
        
        # Detect device
        self.device = next(two_tower_model.parameters()).device
        
        # Set models to eval mode
        self.two_tower_model.eval()
        self.deep_ranking_model.eval()
    
    def build_deal_index(self, all_deal_ids: np.ndarray) -> None:
        """
        Build FAISS index from all deal embeddings.
        
        This pre-computes deal embeddings for fast retrieval.
        """
        check_faiss_available()
        
        print("Computing deal embeddings...")
        
        # Compute all deal embeddings
        batch_size = self.settings.inference.batch_size
        all_embeddings = []
        
        with torch.no_grad():
            for start_idx in range(0, len(all_deal_ids), batch_size):
                end_idx = min(start_idx + batch_size, len(all_deal_ids))
                batch_deals = all_deal_ids[start_idx:end_idx]
                
                deal_data = {
                    "id": torch.tensor(batch_deals, device=self.device),
                    "sector": torch.tensor(
                        [self.deal_features["sector"][d] for d in batch_deals],
                        device=self.device
                    ),
                    "stage": torch.tensor(
                        [self.deal_features["stage"][d] for d in batch_deals],
                        device=self.device
                    ),
                    "region": torch.tensor(
                        [self.deal_features["region"][d] for d in batch_deals],
                        device=self.device
                    ),
                    "deal_size": torch.tensor(
                        [self.deal_features["deal_size"][d] for d in batch_deals],
                        dtype=torch.float32, device=self.device
                    ),
                    "revenue_multiple": torch.tensor(
                        [self.deal_features["revenue_multiple"][d] for d in batch_deals],
                        dtype=torch.float32, device=self.device
                    ),
                    "growth_rate": torch.tensor(
                        [self.deal_features["growth_rate"][d] for d in batch_deals],
                        dtype=torch.float32, device=self.device
                    ),
                    "profitability": torch.tensor(
                        [self.deal_features["profitability"][d] for d in batch_deals],
                        dtype=torch.float32, device=self.device
                    ),
                    "team_experience": torch.tensor(
                        [self.deal_features["team_experience"][d] for d in batch_deals],
                        dtype=torch.float32, device=self.device
                    ),
                    "market_size": torch.tensor(
                        [self.deal_features["market_size"][d] for d in batch_deals],
                        dtype=torch.float32, device=self.device
                    ),
                }
                
                embeddings = self.two_tower_model.encode_deal(deal_data)
                all_embeddings.append(embeddings.cpu().numpy())
        
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        
        # Build FAISS index
        embedding_dim = all_embeddings.shape[1]
        self.faiss_index = FAISSIndex(
            embedding_dim=embedding_dim,
            index_type=self.settings.inference.faiss_index_type,
        )
        self.faiss_index.build(all_embeddings, all_deal_ids)
        
        print(f"Built FAISS index with {len(all_deal_ids)} deals")
    
    def recommend(
        self,
        investor_id: int,
        k: int = 10,
        retrieval_k: int = 100,
        exclude_deal_ids: Optional[set[int]] = None,
    ) -> RecommendationResult:
        """
        Get recommendations for an investor.
        
        Args:
            investor_id: ID of the investor
            k: Number of final recommendations
            retrieval_k: Number of candidates to retrieve before re-ranking
            exclude_deal_ids: Deal IDs to exclude (e.g., already seen)
            
        Returns:
            RecommendationResult with deal IDs and scores
        """
        if self.faiss_index is None:
            raise RuntimeError("FAISS index not built. Call build_deal_index() first.")
        
        exclude_deal_ids = exclude_deal_ids or set()
        
        # Stage 1: Encode investor and retrieve candidates
        with torch.no_grad():
            investor_data = self._prepare_investor_data(investor_id, batch_size=1)
            investor_embedding = self.two_tower_model.encode_investor(investor_data)
            investor_embedding = investor_embedding.cpu().numpy()
        
        # Search FAISS (get more than needed to filter exclusions)
        candidate_ids, retrieval_scores = self.faiss_index.search(
            investor_embedding,
            k=retrieval_k + len(exclude_deal_ids),
            nprobe=self.settings.inference.faiss_nprobe,
        )
        
        # Filter excluded deals
        mask = ~np.isin(candidate_ids, list(exclude_deal_ids))
        candidate_ids = candidate_ids[mask][:retrieval_k]
        retrieval_scores = retrieval_scores[mask][:retrieval_k]
        
        if len(candidate_ids) == 0:
            return RecommendationResult(
                investor_id=investor_id,
                recommended_deal_ids=[],
                scores=[],
                retrieval_candidates=0,
            )
        
        # Stage 2: Re-rank with deep ranking model
        with torch.no_grad():
            batch_size = len(candidate_ids)
            investor_data = self._prepare_investor_data(investor_id, batch_size)
            deal_data = self._prepare_deal_data(candidate_ids)
            
            ranking_scores = self.deep_ranking_model(investor_data, deal_data)
            ranking_scores = ranking_scores.squeeze(-1).cpu().numpy()
        
        # Sort by ranking score
        sorted_indices = np.argsort(-ranking_scores)[:k]
        final_deal_ids = candidate_ids[sorted_indices].tolist()
        final_scores = ranking_scores[sorted_indices].tolist()
        
        return RecommendationResult(
            investor_id=investor_id,
            recommended_deal_ids=final_deal_ids,
            scores=final_scores,
            retrieval_candidates=len(candidate_ids),
        )
    
    def _prepare_investor_data(
        self, investor_id: int, batch_size: int
    ) -> dict[str, torch.Tensor]:
        """Prepare investor data tensors."""
        return {
            "id": torch.tensor([investor_id] * batch_size, device=self.device),
            "type": torch.tensor(
                [self.investor_features["type"][investor_id]] * batch_size,
                device=self.device
            ),
            "region": torch.tensor(
                [self.investor_features["region"][investor_id]] * batch_size,
                device=self.device
            ),
            "risk": torch.tensor(
                [self.investor_features["risk"][investor_id]] * batch_size,
                device=self.device
            ),
            "min_investment": torch.tensor(
                [self.investor_features["min_investment"][investor_id]] * batch_size,
                dtype=torch.float32, device=self.device
            ),
            "max_investment": torch.tensor(
                [self.investor_features["max_investment"][investor_id]] * batch_size,
                dtype=torch.float32, device=self.device
            ),
            "experience_years": torch.tensor(
                [self.investor_features["experience_years"][investor_id]] * batch_size,
                dtype=torch.float32, device=self.device
            ),
            "portfolio_size": torch.tensor(
                [self.investor_features["portfolio_size"][investor_id]] * batch_size,
                dtype=torch.float32, device=self.device
            ),
        }
    
    def _prepare_deal_data(self, deal_ids: np.ndarray) -> dict[str, torch.Tensor]:
        """Prepare deal data tensors."""
        return {
            "id": torch.tensor(deal_ids, device=self.device),
            "sector": torch.tensor(
                [self.deal_features["sector"][d] for d in deal_ids],
                device=self.device
            ),
            "stage": torch.tensor(
                [self.deal_features["stage"][d] for d in deal_ids],
                device=self.device
            ),
            "region": torch.tensor(
                [self.deal_features["region"][d] for d in deal_ids],
                device=self.device
            ),
            "deal_size": torch.tensor(
                [self.deal_features["deal_size"][d] for d in deal_ids],
                dtype=torch.float32, device=self.device
            ),
            "revenue_multiple": torch.tensor(
                [self.deal_features["revenue_multiple"][d] for d in deal_ids],
                dtype=torch.float32, device=self.device
            ),
            "growth_rate": torch.tensor(
                [self.deal_features["growth_rate"][d] for d in deal_ids],
                dtype=torch.float32, device=self.device
            ),
            "profitability": torch.tensor(
                [self.deal_features["profitability"][d] for d in deal_ids],
                dtype=torch.float32, device=self.device
            ),
            "team_experience": torch.tensor(
                [self.deal_features["team_experience"][d] for d in deal_ids],
                dtype=torch.float32, device=self.device
            ),
            "market_size": torch.tensor(
                [self.deal_features["market_size"][d] for d in deal_ids],
                dtype=torch.float32, device=self.device
            ),
        }


def load_inference_pipeline(
    two_tower_checkpoint: str | Path,
    deep_ranking_checkpoint: str | Path,
    encoder_path: str | Path,
    investor_features: dict[str, np.ndarray],
    deal_features: dict[str, np.ndarray],
    settings: Optional[Settings] = None,
) -> RecommenderInference:
    """
    Load models and create inference pipeline.
    
    Args:
        two_tower_checkpoint: Path to Two-Tower model checkpoint
        deep_ranking_checkpoint: Path to Deep Ranking model checkpoint
        encoder_path: Path to saved FeatureEncoder
        investor_features: Dict of investor features
        deal_features: Dict of deal features
        settings: Configuration settings
        
    Returns:
        RecommenderInference pipeline
    """
    if settings is None:
        settings = get_settings()
    
    # Load encoder
    encoder = FeatureEncoder.load(encoder_path)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load models
    two_tower_model = TwoTowerModel.load_from_checkpoint(
        str(two_tower_checkpoint),
        n_investors=len(investor_features["type"]),
        n_deals=len(deal_features["sector"]),
        feature_dims=encoder.feature_dims,
    )
    two_tower_model = two_tower_model.to(device)
    two_tower_model.eval()
    
    deep_ranking_model = PairwiseDeepRankingModel.load_from_checkpoint(
        str(deep_ranking_checkpoint),
        n_investors=len(investor_features["type"]),
        n_deals=len(deal_features["sector"]),
        feature_dims=encoder.feature_dims,
    )
    deep_ranking_model = deep_ranking_model.to(device)
    deep_ranking_model.eval()
    
    return RecommenderInference(
        two_tower_model=two_tower_model,
        deep_ranking_model=deep_ranking_model,
        encoder=encoder,
        investor_features=investor_features,
        deal_features=deal_features,
        settings=settings,
    )
