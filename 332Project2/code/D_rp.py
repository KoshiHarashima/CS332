# Research Payoffs with Fixed Cluster Structure and Regime Switching

import numpy as np
from math import erf, sqrt
from typing import List, Optional

def _phi(x: float) -> float:
    """Standard normal CDF Φ(x) (without SciPy)."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

class ResearchPayoffs:
    """
    Research Payoffs environment with fixed cluster structure and regime switching.
    
    Fixed cluster structure:
    - MIT = {0, 1, 2}
    - Northwestern = {3, 4, 5}  
    - Stanford = {6, 7, 8, 9}
    
    Three regimes with different leading clusters:
    - Regime 0: MIT leading (action 1 is optimal)
    - Regime 1: Northwestern leading (action 4 is optimal)
    - Regime 2: Stanford leading (action 6 is optimal)
    """

    def __init__(
        self,
        k: int = 10,
        mu: float = 0.5,
        delta: float = 0.3,
        rho: float = 0.5,
        h: float = 0.1,
        sigma: float = 0.1,
        seed: Optional[int] = None,
    ):
        """
        Args:
            k: number of actions (papers) = 10
            mu: base mean for coefficients
            delta: regime boost parameter for leader actions
            rho: intra-cluster correlation parameter
            h: regime switching probability
            sigma: noise standard deviation
            seed: RNG seed
        """
        self.k = k
        self.mu = mu
        self.delta = delta
        self.rho = rho
        self.h = h
        self.sigma = sigma
        
        if seed is not None:
            np.random.seed(seed)
        
        # Fixed cluster structure
        self.clusters = {
            'MIT': [0, 1, 2],
            'Northwestern': [3, 4, 5], 
            'Stanford': [6, 7, 8, 9]
        }
        
        # Leader actions for each regime
        self.leaders = {
            0: 1,  # MIT regime -> action 1
            1: 4,  # Northwestern regime -> action 4
            2: 6   # Stanford regime -> action 6
        }
        
        # Current regime state
        self.current_regime = 0
        
        # Cache for covariance matrices
        self._Sigma_cache = {}

    def reset(self):
        """Reset the environment to initial state."""
        self.current_regime = 0

    def _equicorr_cov(self, n: int):
        """Equicorrelation covariance (n x n) with diag=1, offdiag=rho; cached by n."""
        if n <= 1:
            return None
        if n not in self._Sigma_cache:
            # PD check: rho ∈ (-1/(n-1), 1)
            if not (-1.0 / (n - 1) < self.rho < 1.0):
                raise ValueError(
                    f"rho={self.rho} invalid for cluster size n={n}. Needs (-1/(n-1), 1)."
                )
            J = np.ones((n, n))
            I = np.eye(n)
            Sigma = (1 - self.rho) * I + self.rho * J
            self._Sigma_cache[n] = Sigma
        return self._Sigma_cache[n]

    def _sample_cluster_uniforms(self, n: int) -> np.ndarray:
        """
        Sample correlated Uniform[0,1]^n from a Gaussian copula with equicorrelation rho.
        """
        if n == 1:
            return np.random.random(1)
        Sigma = self._equicorr_cov(n)
        z = np.random.multivariate_normal(mean=np.zeros(n), cov=Sigma)  # ~ N(0, Sigma)
        u = np.array([_phi(zi) for zi in z])  # Φ(z) -> Uniform[0,1]
        return u

    def _switch_regime(self):
        """Switch regime with probability h."""
        if np.random.random() < self.h:
            # Switch to a different regime
            other_regimes = [r for r in range(3) if r != self.current_regime]
            self.current_regime = np.random.choice(other_regimes)

    def generate_payoffs(self, round_num: int) -> np.ndarray:
        """
        Generate payoff coefficients for the current round.
        
        Args:
            round_num: current round number (unused but kept for compatibility)
            
        Returns:
            np.array of shape (k,) - coefficients α_{i,j}
        """
        # Switch regime with probability h
        self._switch_regime()
        
        # Initialize coefficients
        alpha = np.zeros(self.k)
        
        # Generate coefficients for each cluster
        for cluster_name, cluster_actions in self.clusters.items():
            n = len(cluster_actions)
            
            # Sample correlated uniforms within cluster
            cluster_uniforms = self._sample_cluster_uniforms(n)
            
            # Convert to coefficients with base mean and regime boost
            for i, action in enumerate(cluster_actions):
                # Base coefficient: μ + η where η ~ N(0, σ²)
                base_coeff = self.mu + np.random.normal(0, self.sigma)
                
                # Add regime boost if this is the leader action
                if action == self.leaders[self.current_regime]:
                    base_coeff += self.delta
                
                # Convert to [0, 1] range using the correlated uniform
                alpha[action] = np.clip(base_coeff, 0, 1)
        
        return alpha

    def compute_payoff(self, weights: np.ndarray, alpha: np.ndarray) -> float:
        """
        Compute the actual payoff according to the mathematical setting.
        
        Args:
            weights: np.array of shape (k,) - allocation weights w_i
            alpha: np.array of shape (k,) - coefficients α_{i,j}
            
        Returns:
            float: Total payoff U_i(w_i) = Σ_j α_{i,j}w_{i,j}
        """
        # Linear payoff: U_i(w_i) = Σ_j α_{i,j}w_{i,j}
        linear_payoff = np.dot(alpha, weights)
        
        return linear_payoff

    def get_current_regime(self) -> int:
        """Get the current regime."""
        return self.current_regime

    def get_leader_action(self) -> int:
        """Get the leader action for the current regime."""
        return self.leaders[self.current_regime]

    def get_cluster_info(self) -> dict:
        """Get information about clusters and current regime."""
        return {
            'current_regime': self.current_regime,
            'leader_action': self.leaders[self.current_regime],
            'clusters': self.clusters
        }