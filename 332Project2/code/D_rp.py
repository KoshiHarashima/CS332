# Research Payoffs with Cluster-based Paper Selection and Markov Regime Switching

import numpy as np
from typing import List, Optional, Dict, Tuple

class ResearchPayoffs:
    """
    Research Payoffs environment with cluster-based paper selection and Markov regime switching.
    
    Key features:
    - Multiple clusters (3 or more)
    - Researcher selects N papers from clusters each round
    - Only one cluster is High at a time
    - Markov regime switching between clusters
    - Paper values follow different uniform distributions based on cluster classification
    """

    def __init__(
        self,
        num_clusters: int = 3,
        papers_per_cluster: int = 10,
        total_papers: int = None,
        high_value_range: Tuple[float, float] = (0.6, 1.0),
        middle_value_range: Tuple[float, float] = (0.4, 0.8),
        low_value_range: Tuple[float, float] = (0.0, 0.5),
        regime_switch_prob: float = 0.3,
        seed: Optional[int] = None,
    ):
        """
        Initialize Research Payoffs environment.
        
        Args:
            num_clusters: number of clusters (must be >= 3)
            papers_per_cluster: number of papers per cluster
            total_papers: total number of papers (if None, calculated as num_clusters * papers_per_cluster)
            high_value_range: value range for High papers (min, max)
            middle_value_range: value range for Middle papers (min, max)
            low_value_range: value range for Low papers (min, max)
            regime_switch_prob: probability of regime switching
            seed: RNG seed
        """
        if num_clusters < 3:
            raise ValueError("Number of clusters must be at least 3")
        
        if papers_per_cluster < 3:
            raise ValueError("Number of papers per cluster must be at least 3 (need 3 papers per cluster for selection)")
            
        self.num_clusters = num_clusters
        self.papers_per_cluster = papers_per_cluster
        self.total_papers = total_papers or (num_clusters * papers_per_cluster)
        self.papers_per_round = num_clusters * 3  # Fixed: 3 papers per cluster
        
        # Value ranges for different paper classifications
        self.high_value_range = high_value_range
        self.middle_value_range = middle_value_range
        self.low_value_range = low_value_range
        
        # Regime switching parameters
        self.regime_switch_prob = regime_switch_prob
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize cluster structure
        self.clusters = self._initialize_clusters()
        
        # Current regime (which cluster is High)
        self.current_regime = 0
        
        # Markov transition matrix (uniform transition probabilities)
        self.transition_matrix = self._create_transition_matrix()
        
        # Current round's selected papers
        self.current_papers = []

    def _initialize_clusters(self) -> Dict[int, List[int]]:
        """Initialize cluster structure."""
        clusters = {}
        for i in range(self.num_clusters):
            start_idx = i * self.papers_per_cluster
            end_idx = start_idx + self.papers_per_cluster
            clusters[i] = list(range(start_idx, end_idx))
        return clusters

    def _create_transition_matrix(self) -> np.ndarray:
        """Create Markov transition matrix for regime switching."""
        # Uniform transition probabilities
        transition_prob = self.regime_switch_prob / (self.num_clusters - 1)
        
        matrix = np.full((self.num_clusters, self.num_clusters), transition_prob)
        
        # Diagonal elements (stay in same regime)
        for i in range(self.num_clusters):
            matrix[i, i] = 1.0 - self.regime_switch_prob
            
        return matrix

    def _switch_regime(self):
        """Switch regime according to Markov transition matrix."""
        # Sample next regime based on current regime
        next_regime = np.random.choice(
            self.num_clusters,
            p=self.transition_matrix[self.current_regime]
        )
        self.current_regime = next_regime

    def _select_papers_from_clusters(self) -> List[int]:
        """
        Select 3 papers from each cluster for the current round.
        Fixed selection with cluster-aware indexing:
        - Papers 0-2: from cluster 0
        - Papers 3-5: from cluster 1
        - Papers 6-8: from cluster 2
        etc.
        """
        selected_papers = []
        
        # Select exactly 3 papers from each cluster with fixed indexing
        for cluster_id in range(self.num_clusters):
            cluster_papers = self.clusters[cluster_id]
            # Randomly select 3 papers from this cluster
            selected_from_cluster = np.random.choice(
                cluster_papers, 
                size=min(3, len(cluster_papers)), 
                replace=False
            ).tolist()
            selected_papers.extend(selected_from_cluster)
        
        return selected_papers

    def _get_paper_value(self, paper_id: int) -> float:
        """Get the value of a paper based on its cluster's regime status."""
        # Find which cluster this paper belongs to
        paper_cluster = None
        for cluster_id, papers in self.clusters.items():
            if paper_id in papers:
                paper_cluster = cluster_id
                break
        
        if paper_cluster is None:
            raise ValueError(f"Paper {paper_id} not found in any cluster")
        
        # Determine cluster regime status
        if paper_cluster == self.current_regime:
            # This cluster is currently High
            value = np.random.uniform(*self.high_value_range)
        elif paper_cluster == (self.current_regime + 1) % self.num_clusters:
            # This cluster is Middle (next cluster in sequence)
            value = np.random.uniform(*self.middle_value_range)
        else:
            # This cluster is Low (all other clusters)
            value = np.random.uniform(*self.low_value_range)
        
        return value

    def generate_payoffs(self, round_num: int) -> Tuple[np.ndarray, List[int]]:
        """
        Generate payoff coefficients for the current round.
        
        Args:
            round_num: current round number (unused but kept for compatibility)
            
        Returns:
            tuple: (alpha_array, selected_papers)
                - alpha_array: np.array of shape (total_papers,) - coefficients α_{i,j}
                - selected_papers: List[int] - papers selected for this round
        """
        # Switch regime according to Markov process
        self._switch_regime()
        
        # Select papers from clusters
        self.current_papers = self._select_papers_from_clusters()
        
        # Generate values for all papers
        alpha = np.zeros(self.total_papers)
        
        for paper_id in range(self.total_papers):
            alpha[paper_id] = self._get_paper_value(paper_id)
        
        return alpha, self.current_papers
    
    def compute_payoff(self, weights: np.ndarray, alpha: np.ndarray, selected_papers: List[int]) -> float:
        """
        Compute the actual payoff according to the mathematical setting.
        
        Args:
            weights: np.array of shape (papers_per_round,) - allocation weights for selected papers
            alpha: np.array of shape (total_papers,) - coefficients α_{i,j}
            selected_papers: List[int] - papers selected for this round
            
        Returns:
            float: Total payoff U_i(w_i) = Σ_j α_{i,j}w_{i,j} for selected papers
        """
        # Linear payoff: U_i(w_i) = Σ_j α_{i,j}w_{i,j} for selected papers only
        selected_alpha = np.array([alpha[paper_id] for paper_id in selected_papers])
        linear_payoff = np.dot(selected_alpha, weights)
        
        return linear_payoff

    def reset(self):
        """Reset the environment to initial state."""
        self.current_regime = 0
        self.current_papers = []

    def get_current_regime(self) -> int:
        """Get the current regime (which cluster is High)."""
        return self.current_regime

    def get_current_papers(self) -> List[int]:
        """Get papers selected for the current round."""
        return self.current_papers.copy()

    def get_cluster_info(self) -> Dict:
        """Get information about clusters and current regime."""
        return {
            'current_regime': self.current_regime,
            'clusters': self.clusters,
            'high_cluster': self.current_regime,
            'middle_cluster': (self.current_regime + 1) % self.num_clusters,
            'low_clusters': [i for i in range(self.num_clusters) 
                           if i not in [self.current_regime, (self.current_regime + 1) % self.num_clusters]],
            'current_papers': self.current_papers
        }

    def get_high_papers(self) -> List[int]:
        """Get papers that are currently High (in the High cluster)."""
        high_cluster = self.current_regime
        return self.clusters[high_cluster]

    def get_middle_papers(self) -> List[int]:
        """Get papers that are currently Middle (in the Middle cluster)."""
        middle_cluster = (self.current_regime + 1) % self.num_clusters
        return self.clusters[middle_cluster]

    def get_low_papers(self) -> List[int]:
        """Get papers that are currently Low (in the Low clusters)."""
        low_papers = []
        for cluster_id in range(self.num_clusters):
            if cluster_id not in [self.current_regime, (self.current_regime + 1) % self.num_clusters]:
                low_papers.extend(self.clusters[cluster_id])
        return low_papers

    def get_paper_classification(self, paper_id: int) -> str:
        """Get the classification of a specific paper based on its cluster's regime status."""
        # Find which cluster this paper belongs to
        for cluster_id, papers in self.clusters.items():
            if paper_id in papers:
                if cluster_id == self.current_regime:
                    return 'high'
                elif cluster_id == (self.current_regime + 1) % self.num_clusters:
                    return 'middle'
                else:
                    return 'low'
        return 'unknown'

    def get_cluster_mapping(self) -> Dict[int, int]:
        """
        Get mapping from paper index to cluster ID for selected papers.
        Returns: {paper_index: cluster_id}
        """
        mapping = {}
        for cluster_id in range(self.num_clusters):
            start_idx = cluster_id * 3
            end_idx = start_idx + 3
            for paper_idx in range(start_idx, end_idx):
                mapping[paper_idx] = cluster_id
        return mapping

    def get_cluster_info_for_researcher(self) -> Dict:
        """
        Get cluster information that the researcher can use for learning.
        """
        return {
            'num_clusters': self.num_clusters,
            'papers_per_cluster': 3,  # Fixed: 3 papers per cluster
            'total_selected_papers': self.num_clusters * 3,
            'cluster_mapping': self.get_cluster_mapping(),
            'current_regime': self.current_regime,
            'high_cluster': self.current_regime,
            'middle_cluster': (self.current_regime + 1) % self.num_clusters,
            'low_clusters': [i for i in range(self.num_clusters) 
                           if i not in [self.current_regime, (self.current_regime + 1) % self.num_clusters]]
        }