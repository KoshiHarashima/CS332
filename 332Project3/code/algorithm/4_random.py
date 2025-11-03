# 4_random.py
# Random algorithm: uniform random bid between 0 and v
# This is equivalent to 3_flexible with epsilon = 0
import numpy as np
from typing import List, Tuple

def random_algorithm(player_id: int, value: float, round_num: int,
                     history: List[Tuple[float, float, bool]],
                     env_state: dict) -> float:
    """
    Random algorithm: bid uniformly random value between 0 and v
    
    Args:
        player_id: player ID (0 or 1)
        value: player's value
        round_num: current round number
        history: list of (bid, utility, won) tuples
        env_state: additional environment state
    
    Returns:
        bid: random value between 0 and value
    """
    return np.random.uniform(0, value)

