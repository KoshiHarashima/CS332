# 3_flexible.py
# Exponential Weight algorithm with learning rate = sqrt(log(k) / n)
# Can also be used with different learning rates (e.g., learning_rate=100 for FTL)
import numpy as np
from typing import List, Tuple

def flexible_algorithm(player_id: int, value: float, round_num: int,
                      history: List[Tuple[float, float, bool]],
                      env_state: dict) -> float:
    """
    Flexible algorithm: Exponential Weight algorithm
    
    Args:
        player_id: player ID (0 or 1)
        value: player's value (v)
        round_num: current round number (n)
        history: list of (bid, utility, won) tuples for this player
        env_state: dict with additional info
            - k: number of arms (discretization), default=100
            - h: scaling parameter, default=value
            - learning_rate: learning rate epsilon, default=sqrt(log(k) / n)
    
    Returns:
        bid: selected bid
    """
    # Get parameters from env_state
    k = env_state.get('k', 100)  # number of arms
    h = env_state.get('h', value)  # scaling parameter
    learning_rate = env_state.get('learning_rate', None)
    
    # Default learning rate: sqrt(log(k) / n)
    if learning_rate is None:
        learning_rate = np.sqrt(np.log(k) / max(round_num, 1))
    
    # Create bid grid (k arms from 0 to value)
    bid_grid = np.linspace(0, value, k)
    
    # Initialize cumulative payoffs for each arm
    if 'cumulative_payoffs' not in env_state:
        env_state['cumulative_payoffs'] = np.zeros(k)
    
    cumulative_payoffs = env_state['cumulative_payoffs']
    
    # If first round, initialize and return middle bid (discretized)
    if round_num == 0 or len(history) == 0:
        bid_grid = np.linspace(0, value, k)
        target_bid = value * 0.5
        discrete_bid_idx = np.argmin(np.abs(bid_grid - target_bid))
        return bid_grid[discrete_bid_idx]
    
    # Update cumulative payoffs from last round's result
    if len(history) > 0:
        last_bid, last_utility, last_won = history[-1]
        # Find which arm (bid) was used
        arm_idx = np.argmin(np.abs(bid_grid - last_bid))
        # Update cumulative payoff for this arm
        cumulative_payoffs[arm_idx] += last_utility
    
    # Exponential Weight selection
    # π_j = (1+ε)^(V_j/h) / Σ_j'(1+ε)^(V_j'/h)
    # where V_j = cumulative_payoffs[j]
    if learning_rate == 0:
        # Random selection if learning_rate is 0
        arm_idx = np.random.randint(0, k)
    else:
        # Calculate probabilities using exponential weights
        powers = (1 + learning_rate) ** (cumulative_payoffs / h)
        sum_powers = np.sum(powers)
        
        if not np.isfinite(sum_powers) or sum_powers <= 0:
            # Fallback to random
            arm_idx = np.random.randint(0, k)
        else:
            probabilities = powers / sum_powers
            if not np.all(np.isfinite(probabilities)):
                arm_idx = np.random.randint(0, k)
            else:
                arm_idx = np.random.choice(k, p=probabilities)
    
    return bid_grid[arm_idx]

