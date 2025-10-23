# Fixed-Share Exponentiated Gradient Algorithm for Research Payoffs

import numpy as np
from typing import Callable, Tuple

class FixedShareEG:
    """
    Fixed-Share Exponentiated Gradient Algorithm for Online Linear Optimization.
    
    This algorithm is designed for environments with regime switching where
    the optimal action may change over time.
    
    Update rule:
    1. w̃_t = (1 - λ) w_t + λ · 1/k  (Fixed-Share update)
    2. w_{t+1} ∝ w̃_t ⊙ exp(ε · α_t)  (Exponentiated Gradient update)
    """

    def __init__(self, k: int, epsilon: float, n: int, lambda_param: float = None):
        """
        Initialize Fixed-Share Exponentiated Gradient algorithm.
        
        Args:
            k: number of actions
            epsilon: learning rate parameter
            n: total number of rounds
            lambda_param: fixed-share parameter (if None, will be set to 1/L where L is average regime length)
        """
        self.k = k
        self.epsilon = epsilon
        self.n = n
        
        # Set lambda parameter
        if lambda_param is None:
            # Default: lambda ≈ 1/L where L is average regime length
            # Assuming regime switching probability h, average regime length L = 1/h
            # So lambda = h
            self.lambda_param = 0.1  # Default for h = 0.1
        else:
            self.lambda_param = lambda_param
        
        # Initialize weights uniformly
        self.weights = np.ones(k) / k
        
        # History tracking
        self.regret_history = []
        self.total_payoff = 0
        self.payoff_history = []
        self.weight_history = []
        self.cumulative_alpha = np.zeros(k)

    def select_weights(self) -> np.ndarray:
        """
        Select weights for the current round.
        
        Returns:
            np.array of shape (k,) - current weight allocation
        """
        return self.weights.copy()

    def update_weights(self, alpha: np.ndarray):
        """
        Update weights using Fixed-Share Exponentiated Gradient update.
        
        Args:
            alpha: np.array of shape (k,) - coefficients α_{i,j} for current round
        """
        # Compute current payoff: U_i(w_i) = Σ_j α_{i,j}w_{i,j}
        current_payoff = np.dot(alpha, self.weights)
        
        # Update total payoff
        self.total_payoff += current_payoff
        
        # Update cumulative alpha for regret calculation
        self.cumulative_alpha += alpha
        
        # Fixed-Share Exponentiated Gradient update
        # Step 1: Fixed-Share update
        # w̃_t = (1 - λ) w_t + λ · 1/k
        uniform_weights = np.ones(self.k) / self.k
        w_tilde = (1 - self.lambda_param) * self.weights + self.lambda_param * uniform_weights
        
        # Step 2: Exponentiated Gradient update
        # w_{t+1} ∝ w̃_t ⊙ exp(ε · α_t)
        unnormalized_weights = w_tilde * np.exp(self.epsilon * alpha)
        
        # Normalize to ensure weights sum to 1
        self.weights = unnormalized_weights / np.sum(unnormalized_weights)
        
        # Compute cumulative regret
        # Regret = max_j(cumulative_alpha_j) - total_payoff
        optimal_cumulative_payoff = np.max(self.cumulative_alpha)
        cumulative_regret = optimal_cumulative_payoff - self.total_payoff
        
        # Track history
        self.regret_history.append(cumulative_regret)
        self.payoff_history.append(self.total_payoff)
        self.weight_history.append(self.weights.copy())

    def run_algorithm(self, payoff_generator: Callable[[int], np.ndarray]) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        """
        Run the algorithm for n rounds.
        
        Args:
            payoff_generator: Function that generates coefficients α_{i,j}
            
        Returns:
            tuple: (regret_history, total_payoff, cumulative_payoffs, payoff_history)
        """
        cumulative_payoffs = np.zeros(self.k)
        
        for round_num in range(self.n):
            # Generate coefficients
            alpha = payoff_generator(round_num)
            
            # Select weights
            weights = self.select_weights()
            
            # Compute payoff
            payoff = np.dot(alpha, weights)
            
            # Update cumulative payoffs for each action
            cumulative_payoffs += alpha
            
            # Update weights
            self.update_weights(alpha)
        
        return (
            np.array(self.regret_history),
            self.total_payoff,
            cumulative_payoffs,
            np.array(self.payoff_history)
        )

    def get_current_weights(self) -> np.ndarray:
        """Get current weight allocation."""
        return self.weights.copy()

    def get_regret_history(self) -> np.ndarray:
        """Get regret history."""
        return np.array(self.regret_history)

    def get_payoff_history(self) -> np.ndarray:
        """Get payoff history."""
        return np.array(self.payoff_history)

    def get_weight_history(self) -> np.ndarray:
        """Get weight history."""
        return np.array(self.weight_history)

    def reset(self):
        """Reset the algorithm to initial state."""
        self.weights = np.ones(self.k) / k
        self.regret_history = []
        self.total_payoff = 0
        self.payoff_history = []
        self.weight_history = []
        self.cumulative_alpha = np.zeros(self.k)
