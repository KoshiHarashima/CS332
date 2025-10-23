# Exponentiated Gradient Algorithm for Research Payoffs
# This implements the mathematical setting where researcher allocates weights to all actions

import numpy as np


class ExponentiatedGradient:
    """
    Exponentiated Gradient Algorithm for Research Payoffs.
    
    This algorithm implements the mathematical setting where:
    - Researcher allocates weights w_i ∈ Δ^N to all actions
    - Linear payoff: U_i(w_i) = Σ_j α_{i,j}w_{i,j}
    - Uses gradient-based updates: w_{t+1} = w_t * exp(ε * α_t) / Z_t
    """
    
    def __init__(self, k, epsilon, n):
        """
        Initialize the Exponentiated Gradient algorithm.
        
        Args:
            k: Number of actions
            epsilon: Learning rate (step size)
            n: Number of rounds
        """
        self.k = k
        self.epsilon = epsilon
        self.n = n
        
        # Initialize uniform weights
        self.weights = np.ones(k) / k
        
        # Track history
        self.regret_history = []
        self.total_payoff = 0
        self.payoff_history = []
        self.weight_history = []  # Track weight evolution
        
    def select_weights(self):
        """
        Return current weight allocation.
        This represents the researcher's time allocation across all actions.
        
        Returns:
            np.array: Weight allocation w_i ∈ Δ^N
        """
        return self.weights.copy()
    
    def update_weights(self, alpha):
        """
        Update weights using exponentiated gradient update.
        
        Args:
            alpha: np.array of shape (k,) - coefficients α_{i,j} for current round
        """
        # Compute current payoff: U_i(w_i) = Σ_j α_{i,j}w_{i,j}
        current_payoff = np.dot(alpha, self.weights)
        
        # Update total payoff
        self.total_payoff += current_payoff
        
        # Exponentiated gradient update
        # w_{t+1} = w_t * exp(ε * α_t) / Z_t
        # where Z_t is the normalization factor
        
        # Compute unnormalized weights
        unnormalized_weights = self.weights * np.exp(self.epsilon * alpha)
        
        # Normalize to ensure weights sum to 1
        self.weights = unnormalized_weights / np.sum(unnormalized_weights)
        
        # Compute cumulative regret (difference from optimal cumulative payoff)
        cumulative_regret = self._compute_cumulative_regret(alpha)
        regret = cumulative_regret
        
        # Track history
        self.regret_history.append(regret)
        self.payoff_history.append(self.total_payoff)
        self.weight_history.append(self.weights.copy())
    
    def _compute_optimal_payoff(self, alpha):
        """
        Compute the optimal payoff using the best possible weight allocation.
        
        For linear payoff U(w) = Σ_j α_j w_j, the optimal strategy is to put
        all weight on the action with the highest coefficient.
        
        Args:
            alpha: np.array of shape (k,) - coefficients α_{i,j}
            
        Returns:
            float: Optimal payoff
        """
        # For linear payoff, the optimal strategy is to put all weight on the best action
        optimal_payoff = np.max(alpha)
        return optimal_payoff
    
    def _compute_cumulative_regret(self, alpha):
        """
        Compute cumulative regret up to current round.
        
        This is the difference between the optimal cumulative payoff and
        the actual cumulative payoff.
        
        Args:
            alpha: np.array of shape (k,) - coefficients α_{i,j} for current round
            
        Returns:
            float: Cumulative regret
        """
        # Update cumulative payoffs for each action
        if not hasattr(self, 'cumulative_alpha'):
            self.cumulative_alpha = np.zeros(self.k)
        self.cumulative_alpha += alpha
        
        # Optimal cumulative payoff (best single action in hindsight)
        optimal_cumulative_payoff = np.max(self.cumulative_alpha)
        
        # Current cumulative payoff
        current_cumulative_payoff = self.total_payoff
        
        # Cumulative regret
        cumulative_regret = optimal_cumulative_payoff - current_cumulative_payoff
        
        return cumulative_regret
    
    def run_algorithm(self, payoff_generator):
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
            
            # Update weights based on coefficients
            self.update_weights(alpha)
            
            # Update cumulative payoffs (for compatibility)
            cumulative_payoffs += alpha
        
        return self.regret_history, self.total_payoff, cumulative_payoffs, self.payoff_history


class ExponentiatedGradientWithProjection:
    """
    Exponentiated Gradient with projection to simplex.
    Alternative implementation that ensures weights stay in the simplex.
    """
    
    def __init__(self, k, epsilon, n):
        self.k = k
        self.epsilon = epsilon
        self.n = n
        
        # Initialize uniform weights
        self.weights = np.ones(k) / k
        
        # Track history
        self.regret_history = []
        self.total_payoff = 0
        self.payoff_history = []
        self.weight_history = []
        
    def select_weights(self):
        """Return current weight allocation."""
        return self.weights.copy()
    
    def _project_to_simplex(self, weights):
        """
        Project weights to the simplex (sum to 1, all non-negative).
        """
        # Sort weights in descending order
        sorted_indices = np.argsort(weights)[::-1]
        sorted_weights = weights[sorted_indices]
        
        # Find the threshold
        cumsum = np.cumsum(sorted_weights)
        threshold = (cumsum - 1) / np.arange(1, self.k + 1)
        
        # Find the largest threshold that keeps weights non-negative
        for i in range(self.k - 1, -1, -1):
            if sorted_weights[i] > threshold[i]:
                break
        
        # Project
        projected_weights = np.maximum(weights - threshold[i], 0)
        return projected_weights / np.sum(projected_weights)
    
    def update_weights(self, alpha):
        """
        Update weights using exponentiated gradient with projection.
        """
        # Compute current payoff
        current_payoff = np.dot(alpha, self.weights)
        self.total_payoff += current_payoff
        
        # Exponentiated gradient update in log-space for stability
        log_weights = np.log(self.weights + 1e-10)  # Add small epsilon for numerical stability
        log_weights += self.epsilon * alpha
        
        # Project back to simplex
        self.weights = self._project_to_simplex(np.exp(log_weights))
        
        # Compute regret
        optimal_payoff = np.max(alpha)
        regret = optimal_payoff - current_payoff
        
        # Track history
        self.regret_history.append(regret)
        self.payoff_history.append(self.total_payoff)
        self.weight_history.append(self.weights.copy())
    
    def run_algorithm(self, payoff_generator):
        """Run the algorithm for n rounds."""
        cumulative_payoffs = np.zeros(self.k)
        
        for round_num in range(self.n):
            alpha = payoff_generator(round_num)
            self.update_weights(alpha)
            cumulative_payoffs += alpha
        
        return self.regret_history, self.total_payoff, cumulative_payoffs, self.payoff_history

