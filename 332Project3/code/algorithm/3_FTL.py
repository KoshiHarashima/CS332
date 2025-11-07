# 3_FTL.py
# Follow The Leader (FTL) algorithm
# 
# Overflow avoidance note:
# This implementation avoids overflow by directly selecting the best arm based on
# cumulative payoffs, rather than using exponential weights with large learning rates.
# The previous approach using learning_rate=100 in EW algorithm caused overflow warnings
# because (1 + learning_rate)^(cumulative_payoffs/h) could produce extremely large values.
# FTL simply selects argmax(cumulative_payoffs), which is numerically stable.
#
# Strategy:
# - At each round, selects the arm with the highest cumulative payoff
# - Uses Full Feedback to update ALL arms' utilities at each round
# - No probabilistic selection (deterministic choice of best arm)

import numpy as np
from typing import List, Tuple

def ftl_algorithm(player_id: int, value: float, round_num: int,
                  history: List[Tuple[float, float, bool, float]],
                  env_state: dict) -> float:
    """
    Follow The Leader (FTL) algorithm
    
    Full Information + Full Feedback: Can observe opponent's bids directly.
    
    Strategy:
    - Uses Full Feedback to update ALL arms' utilities at each round
    - At each round, calculates utility for ALL arms j: u_j = (v - b_j) * Pr(win_j)
      where Pr(win_j) is deterministically determined from opponent_bid (0/1/0.5)
    - Updates cumulative_payoffs[j] += u_j for all arms
    - Selects the arm with maximum cumulative payoff: argmax(cumulative_payoffs)
    
    Overflow avoidance:
    - Directly uses argmax instead of exponential weights
    - Avoids computing (1 + learning_rate)^(cumulative_payoffs/h) which can overflow
    - Numerically stable implementation
    
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
            - h: scaling parameter, default=value (not used in FTL, kept for compatibility)
    
    Returns:
        bid: selected bid
    """
    # Get parameters from env_state
    k = env_state.get('k', 100)  # number of arms
    h = env_state.get('h', value)  # scaling parameter (not used in FTL, kept for compatibility)
    
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
    
    # FTL: Select arm with maximum cumulative payoff among valid arms
    # Overflow avoidance: Direct argmax instead of exponential weights
    # This avoids computing (1 + learning_rate)^(cumulative_payoffs/h) which can overflow
    # Only consider valid bids
    valid_cumulative = cumulative_payoffs.copy()
    valid_cumulative[~valid_mask] = -np.inf  # Set invalid bids to -inf
    
    best_arm_idx = np.argmax(valid_cumulative)
    
    # Get player's independent random state from env_state
    random_state = env_state.get('random_state', np.random)
    
    # Handle ties: if multiple arms have the same max payoff, choose randomly among them
    max_payoff = cumulative_payoffs[best_arm_idx]
    ties = np.where(np.isclose(cumulative_payoffs, max_payoff, rtol=1e-9, atol=1e-9) & valid_mask)[0]
    
    if len(ties) > 1:
        # Multiple ties: choose randomly among them
        best_arm_idx = random_state.choice(ties)
    
    return bid_grid[best_arm_idx]

