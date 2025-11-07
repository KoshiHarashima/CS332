# 2_ew.py
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
    - Uses exponential weights: π_j = exp(ε*V_j/h) / Σ_j'exp(ε*V_j'/h)
    
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
    
    # Use unified_bid_grid if available (for consistency across rounds when values change)
    # Otherwise, create bid_grid based on current value (for backward compatibility)
    if 'unified_bid_grid' in env_state:
        bid_grid = env_state['unified_bid_grid']
        # Only consider bids that are valid (bid <= value) for this round
        valid_mask = bid_grid <= value
        if not np.any(valid_mask):
            # No valid bids, return 0
            return 0.0
    else:
        # Backward compatibility: create bid_grid based on current value
        bid_grid = np.linspace(0, value, k)
        valid_mask = np.ones(len(bid_grid), dtype=bool)  # All bids are valid
    
    # Initialize cumulative payoffs for each arm (size matches bid_grid)
    if 'cumulative_payoffs' not in env_state:
        env_state['cumulative_payoffs'] = np.zeros(len(bid_grid))
    
    cumulative_payoffs = env_state['cumulative_payoffs']
    
    # Ensure cumulative_payoffs size matches bid_grid size
    if len(cumulative_payoffs) != len(bid_grid):
        # Resize cumulative_payoffs to match bid_grid
        old_size = len(cumulative_payoffs)
        new_size = len(bid_grid)
        if new_size > old_size:
            # Expand: pad with zeros
            cumulative_payoffs = np.pad(cumulative_payoffs, (0, new_size - old_size), 'constant')
            env_state['cumulative_payoffs'] = cumulative_payoffs
        else:
            # Shrink: truncate (shouldn't happen with unified_bid_grid)
            cumulative_payoffs = cumulative_payoffs[:new_size]
            env_state['cumulative_payoffs'] = cumulative_payoffs
    
    # If first round, start with b = 0 * v = 0
    if round_num == 0 or len(history) == 0:
        target_bid = 0.0  # 0 * v
        # Find closest valid bid
        valid_bids = bid_grid[valid_mask]
        if len(valid_bids) == 0:
            return 0.0
        discrete_bid_idx = np.argmin(np.abs(valid_bids - target_bid))
        return valid_bids[discrete_bid_idx]
    
    # Full Feedback: Update ALL arms' utilities from last round's opponent_bid
    if len(history) > 0:
        last_entry = history[-1]
        opponent_bid = last_entry[3]  # opponent's bid from Full Feedback
        
        # Calculate utility for ALL arms j: u_j = (v - b_j) * Pr(win_j)
        # where Pr(win_j) is deterministically determined from opponent_bid
        # Vectorized computation for better performance
        
        # Use relative tolerance for tie detection
        rtol = 1e-9
        atol = 1e-9
        
        # Only update utilities for valid bids (bid <= value)
        valid_bid_grid = bid_grid[valid_mask]
        
        # Vectorized allocation calculation
        greater_mask = valid_bid_grid > opponent_bid
        less_mask = valid_bid_grid < opponent_bid
        tie_mask = np.isclose(valid_bid_grid, opponent_bid, rtol=rtol, atol=atol)
        
        # Calculate Pr(win_j) vectorized
        pr_win = np.zeros(len(valid_bid_grid))
        pr_win[greater_mask & ~tie_mask] = 1.0
        pr_win[tie_mask] = 0.5
        pr_win[less_mask & ~tie_mask] = 0.0
        
        # Calculate utilities vectorized: u_j = (v - b_j) * Pr(win_j)
        utilities = (value - valid_bid_grid) * pr_win
        
        # Update cumulative payoffs for valid arms only
        cumulative_payoffs[valid_mask] += utilities
    
    # Exponential Weight selection over valid arms only
    # Create a mask for valid arms with non-negative cumulative payoffs
    valid_cumulative = cumulative_payoffs.copy()
    valid_cumulative[~valid_mask] = -np.inf  # Set invalid bids to -inf
    
    # Get player's independent random state from env_state
    random_state = env_state.get('random_state', np.random)
    
    if learning_rate == 0:
        # Random selection among valid bids
        valid_indices = np.where(valid_mask)[0]
        if len(valid_indices) == 0:
            return 0.0
        arm_idx = random_state.choice(valid_indices)
    else:
        # Calculate probabilities using exponential weights
        # Only consider valid arms
        valid_payoffs = cumulative_payoffs[valid_mask]
        if len(valid_payoffs) == 0:
            return 0.0
        
        # Standard form: exp(ε * V_j / h)
        powers = np.exp(learning_rate * valid_payoffs / h)
        sum_powers = np.sum(powers)
        
        if not np.isfinite(sum_powers) or sum_powers <= 0:
            # Fallback to random among valid bids
            valid_indices = np.where(valid_mask)[0]
            arm_idx = random_state.choice(valid_indices)
        else:
            probabilities = powers / sum_powers
            if not np.all(np.isfinite(probabilities)):
                valid_indices = np.where(valid_mask)[0]
                arm_idx = random_state.choice(valid_indices)
            else:
                # Select from valid bids
                valid_indices = np.where(valid_mask)[0]
                arm_idx = valid_indices[random_state.choice(len(valid_indices), p=probabilities)]
    
    return bid_grid[arm_idx]

