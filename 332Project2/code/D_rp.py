# Research Payoffs with Regime Switching and Cluster Structure
# 
# Mathematical Description:
# In each round i:
# 1. Draw a latent regime state z_i ∈ {1, ..., m} with Markov transition:
#    Pr(z_i = z_{i-1}) = 1-h, Pr(z_i ≠ z_{i-1}) = h
# 2. For each cluster C_c, draw correlated coefficients:
#    α_{i,C_c} ~ N(μ + δ·1{c = z_i}, σ² Σ_c)
#    where Σ_c has intra-cluster correlation ρ
# 3. Writing coefficient: α_{i,0} ~ U[0,1]
# 4. Researcher allocates weights w_i ∈ Δ^N
# 5. Linear payoff: U_i^lin(w_i) = α_{i,0}w_{i,0} + Σ_j α_{i,j}w_{i,j}
# 6. Mismatch penalty: U_i = U_i^lin(w_i) - λ max{0, w_{i,0} - K_i}

import numpy as np

class ResearchPayoffs:
    """
    Research payoffs model with regime switching and cluster structure.
    
    The model simulates a researcher who allocates time between:
    - Writing (action 0): Independent coefficient α_{i,0} ~ U[0,1]
    - Reading papers from different clusters (actions 1 to N): 
      Correlated coefficients with regime-dependent means
    """
    
    def __init__(self, k):
        """
        Initialize the research payoffs model.
        
        Args:
            k: Number of actions (writing + reading clusters)
        """
        self.k = k  # Total actions: 0 (writing) + (k-1) (reading clusters)
        self.m = 3  # Number of regime states (clusters)
        self.N = k - 1  # Number of paper clusters (actions 1 to k-1)
        self.h = 0.1  # Regime change probability
        self.mu = 0.5  # Base mean
        self.delta = 0.3  # Regime effect
        self.sigma = 0.2  # Standard deviation
        self.rho = 0.5  # Intra-cluster correlation
        self.lam = 1.0  # Mismatch penalty parameter
        
        # Current regime state
        self.current_regime = None
        
        # Reading coefficients r_{i,j} (usefulness of reading each cluster)
        # These are fixed parameters representing how useful each cluster is
        self.r = np.random.uniform(0.1, 0.9, self.N)
    
    def _draw_regime_state(self):
        """
        Draw regime state z_i with Markov transition.
        Pr(z_i = z_{i-1}) = 1-h, Pr(z_i ≠ z_{i-1}) = h
        """
        if self.current_regime is None:
            # First round: draw uniformly
            self.current_regime = np.random.randint(1, self.m + 1)
        else:
            # Markov transition
            if np.random.random() < (1 - self.h):
                # Stay in current regime
                pass
            else:
                # Switch to different regime
                other_regimes = [r for r in range(1, self.m + 1) if r != self.current_regime]
                self.current_regime = np.random.choice(other_regimes)
        
        return self.current_regime
    
    def _draw_coefficients(self):
        """
        Draw coefficients α_{i,j} for each cluster.
        α_{i,C_c} ~ N(μ + δ·1{c = z_i}, σ² Σ_c)
        α_{i,0} ~ U[0,1] (writing coefficient)
        """
        z_i = self.current_regime
        
        # Writing coefficient (independent)
        alpha_0 = np.random.uniform(0, 1)
        
        # Reading coefficients for each cluster
        alpha_reading = np.zeros(self.N)
        
        for c in range(1, self.m + 1):
            # Find papers in cluster c
            cluster_papers = list(range(c-1, self.N, self.m))  # Distribute papers across clusters
            
            if cluster_papers:
                n_papers = len(cluster_papers)
                
                # Mean vector: μ + δ·1{c = z_i}
                mean = self.mu + self.delta * (1 if c == z_i else 0)
                
                # Correlation matrix Σ_c
                Sigma_c = np.eye(n_papers) + self.rho * (np.ones((n_papers, n_papers)) - np.eye(n_papers))
                
                # Draw correlated coefficients
                cluster_coeffs = np.random.multivariate_normal(
                    mean * np.ones(n_papers),
                    self.sigma**2 * Sigma_c
                )
                
                # Assign to corresponding papers
                for i, paper_idx in enumerate(cluster_papers):
                    if paper_idx < self.N:
                        alpha_reading[paper_idx] = cluster_coeffs[i]
        
        return alpha_0, alpha_reading
    
    def generate_payoffs(self, round_num):
        """
        Generate payoffs for the current round.
        
        Args:
            round_num: Current round number (not used in this model)
            
        Returns:
            np.array: Payoff for each action [writing, reading_cluster_1, ..., reading_cluster_N]
        """
        # Draw regime state
        z_i = self._draw_regime_state()
        
        # Draw coefficients
        alpha_0, alpha_reading = self._draw_coefficients()
        
        # Combine all coefficients
        alpha = np.concatenate([[alpha_0], alpha_reading])
        
        return alpha
    
    def reset(self):
        """Reset the model state."""
        self.current_regime = None