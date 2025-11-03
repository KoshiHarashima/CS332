# 3_flexible.py
# Exponential Weight algorithm with learning rate = sqrt(log(k) / n)
# Optimal learning rate: epsilon = sqrt(log(k) / T)
# Note: cumulative_payoffs are normalized by h, so epsilon does not need h factor
import numpy as np
from typing import List, Tuple

def flexible_algorithm(player_id: int, value: float, round_num: int,
                      history: List[Tuple[float, float, bool, float]],
                      env_state: dict) -> float:
    """
    Flexible algorithm: Exponential Weight algorithm with Full Feedback
    
    Full Information + Full Feedback: Can observe opponent's bids directly.
    
    Strategy:
    - Uses Full Feedback to update ALL arms' utilities at each round
    - At each round, calculates utility for ALL arms j: u_j = (v - b_j) * Pr(win_j)
      where Pr(win_j) is deterministically determined from opponent_bid (0/1/0.5)
    - Updates cumulative_payoffs[j] += u_j for all arms
    - Uses exponential weights: π_j = (1+ε)^(V_j/h) / Σ_j'(1+ε)^(V_j'/h)
    
    Args:
        player_id: player ID (0 or 1)
        value: player's value (v)
        round_num: current round number (n)
        history: list of (bid, utility, won, opponent_bid) tuples for this player
                 - bid: player's bid
                 - utility: player's utility
                 - won: whether player won
                 - opponent_bid: opponent's bid (Full Feedback)
        env_state: dict with additional info
            - k: number of arms (discretization), default=100
            - h: scaling parameter, default=value
            - learning_rate: learning rate epsilon, default=sqrt(log(k) / n)
    
    Returns:
        bid: selected bid
    """
    # Get parameters from env_state
    k = env_state.get('k', 100)  # number of arms
    h = env_state.get('h', value)  # scaling parameter (payoff range is [0, h])
    learning_rate = env_state.get('learning_rate', None)
    
    # Default learning rate: epsilon = sqrt(log(k) / n)
    # Optimal learning rate for Exponential Weight: epsilon = sqrt(log(k) / T)
    # Note: Since cumulative_payoffs are normalized by h in the weight calculation,
    # the learning rate epsilon does not need the h factor
    if learning_rate is None:
        learning_rate = np.sqrt(np.log(k) / max(round_num, 1))
    
    # Create bid grid (k arms from 0 to value) - unified grid
    bid_grid = np.linspace(0, value, k)
    
    # Initialize cumulative payoffs for each arm
    if 'cumulative_payoffs' not in env_state:
        env_state['cumulative_payoffs'] = np.zeros(k)
    
    cumulative_payoffs = env_state['cumulative_payoffs']
    
    # If first round, initialize and return maximum bid (v * 1.0, discretized)
    if round_num == 0 or len(history) == 0:
        target_bid = value * 1.0
        discrete_bid_idx = np.argmin(np.abs(bid_grid - target_bid))
        return bid_grid[discrete_bid_idx]
    
    # Full Feedback: Update ALL arms' utilities from last round's opponent_bid
    if len(history) > 0:
        last_entry = history[-1]
        opponent_bid = last_entry[3]  # opponent's bid from Full Feedback
        
        # Calculate utility for ALL arms j: u_j = (v - b_j) * Pr(win_j)
        # where Pr(win_j) is deterministically determined from opponent_bid
        utilities = np.zeros(k)
        
        # Use relative tolerance for tie detection
        rtol = 1e-9
        atol = 1e-9
        
        for j, bid_j in enumerate(bid_grid):
            # Determine Pr(win_j) from opponent_bid
            if bid_j > opponent_bid and not np.isclose(bid_j, opponent_bid, rtol=rtol, atol=atol):
                # Win: our bid > opponent_bid
                pr_win = 1.0
            elif bid_j < opponent_bid and not np.isclose(bid_j, opponent_bid, rtol=rtol, atol=atol):
                # Lose: our bid < opponent_bid
                pr_win = 0.0
            else:
                # Tie: our bid == opponent_bid (within tolerance)
                pr_win = 0.5
            
            # Utility: u_j = (v - b_j) * Pr(win_j)
            # Note: In FPA, if win, payment = bid_j, so utility = v - bid_j
            # If lose, utility = 0
            # If tie, utility = (v - bid_j) * 0.5 (expected utility)
            utilities[j] = (value - bid_j) * pr_win
        
        # Update cumulative payoffs for ALL arms
        cumulative_payoffs += utilities
    
    # Exponential Weight selection over k arms (discrete bids)
    # Probability distribution: π_j = (1+ε)^(V_j/h) / Σ_j'(1+ε)^(V_j'/h)
    # where V_j = cumulative_payoffs[j] (cumulative utility from arm j)
    # and ε = learning_rate = sqrt(log(k) / n) by default (optimal)
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

