# 4_uniform_guessing.py
# Uniform Guessing algorithm
#
# Overflow avoidance note:
# This implementation avoids overflow by using simple random selection without
# any exponential weight calculations. The previous approach using learning_rate=0.01
# in EW algorithm was an approximation. This implementation provides true uniform
# random selection, which is computationally efficient and numerically stable.
#
# Strategy:
# - At each round, randomly selects a bid from the bid grid uniformly
# - No learning or adaptation (completely random)
# - Serves as a baseline for comparison

import numpy as np
from typing import List, Tuple

def uniform_guessing_algorithm(player_id: int, value: float, round_num: int,
                               history: List[Tuple[float, float, bool, float]],
                               env_state: dict) -> float:
    """
    Uniform Guessing algorithm: completely random bid selection
    
    Full Information + Full Feedback: Can observe opponent's bids directly.
    However, this algorithm ignores all feedback and selects bids randomly.
    
    Strategy:
    - At each round, randomly selects a bid from the bid grid uniformly
    - No learning or adaptation
    - No use of history or opponent bids
    - Serves as a baseline for comparison
    
    Overflow avoidance:
    - Simple random selection without any exponential calculations
    - No cumulative payoffs or weight calculations needed
    - Numerically stable and computationally efficient
    
    Args:
        player_id: player ID (0 or 1)
        value: player's value (v)
        round_num: current round number (n)
        history: list of (bid, utility, won, opponent_bid) tuples for this player
                 (not used in this algorithm, kept for compatibility)
        env_state: dict with additional info
            - k: number of arms (discretization), default=100
    
    Returns:
        bid: randomly selected bid
    """
    # Get parameters from env_state
    k = env_state.get('k', 100)  # number of arms
    
    # Create bid grid (k arms from 0 to value)
    bid_grid = np.linspace(0, value, k)
    
    # First round: start with b = 0 * v = 0
    if round_num == 0:
        target_bid = 0.0  # 0 * v
        discrete_bid_idx = np.argmin(np.abs(bid_grid - target_bid))
        return bid_grid[discrete_bid_idx]
    
    # Uniform Guessing: Randomly select any bid from the grid
    # Overflow avoidance: Simple random selection, no exponential calculations
    # This avoids any potential overflow from exponential weight computations
    arm_idx = np.random.randint(0, k)
    
    return bid_grid[arm_idx]

