# 4_cool.py
# Cool algorithm: always bids theoretical optimal value v/2
import numpy as np
from typing import List, Tuple

def cool_algorithm(player_id: int, value: float, round_num: int,
                  history: List[Tuple[float, float, bool]],
                  env_state: dict) -> float:
    """
    Cool algorithm: always bids theoretical optimal value v/2 (discretized)
    
    Args:
        player_id: player ID (0 or 1)
        value: player's value
        round_num: current round number
        history: list of (bid, utility, won) tuples
        env_state: additional environment state
    
    Returns:
        bid: discretized value closest to value / 2
    """
    # Discretize: select from k discrete arms
    k = env_state.get('k', 100)
    bid_grid = np.linspace(0, value, k)
    target_bid = value / 2.0
    discrete_bid_idx = np.argmin(np.abs(bid_grid - target_bid))
    return bid_grid[discrete_bid_idx]

