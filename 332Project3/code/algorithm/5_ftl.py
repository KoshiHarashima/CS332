# 5_ftl.py
# Follow The Leader: epsilon = 100 in Exponential Weight
# Note: This assumes 3_flexible is implemented with exponential weights
# For now, we'll use a simple FTL approach: bid the best bid so far
import numpy as np
from typing import List, Tuple

def ftl_algorithm(player_id: int, value: float, round_num: int,
                  history: List[Tuple[float, float, bool]],
                  env_state: dict) -> float:
    """
    Follow The Leader algorithm: choose the bid that performed best so far
    
    Args:
        player_id: player ID (0 or 1)
        value: player's value
        round_num: current round number
        history: list of (bid, utility, won) tuples
        env_state: additional environment state (can set epsilon=100 if using 3_flexible)
    
    Returns:
        bid: the bid that has given the highest cumulative utility so far
    """
    if round_num == 0 or len(history) == 0:
        # First round: start with value/2
        return value * 0.5
    
    # Find the bid that has the highest cumulative utility
    bid_utilities = {}
    for bid, utility, won in history:
        if bid not in bid_utilities:
            bid_utilities[bid] = 0
        bid_utilities[bid] += utility
    
    # Select bid with highest cumulative utility
    if len(bid_utilities) > 0:
        best_bid = max(bid_utilities.items(), key=lambda x: x[1])[0]
        return best_bid
    
    # Fallback
    return value * 0.5

