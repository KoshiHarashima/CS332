# 6_cool.py
# Cool algorithm: always bids theoretical optimal value v/2
import numpy as np
from typing import List, Tuple

def cool_algorithm(player_id: int, value: float, round_num: int,
                  history: List[Tuple[float, float, bool]],
                  env_state: dict) -> float:
    """
    Cool algorithm: always bids 1/2 * value
    
    Args:
        player_id: player ID (0 or 1)
        value: player's value
        round_num: current round number
        history: list of (bid, utility, won) tuples
        env_state: additional environment state
    
    Returns:
        bid: always value / 2
    """
    return value / 2.0

