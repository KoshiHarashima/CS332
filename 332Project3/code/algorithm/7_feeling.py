# 7_feeling.py
# Feeling algorithm: chooses strategy based on past results
# Three strategies:
#   1. Aggressive: 3/4 of v (if losing many times)
#   2. Normal: 1/2 of v (if winning many times)
#   3. Conservative: 1/4 of v (if mixed win/lose)
import numpy as np
from typing import List, Tuple

def feeling_algorithm(player_id: int, value: float, round_num: int,
                      history: List[Tuple[float, float, bool]],
                      env_state: dict) -> float:
    """
    Feeling algorithm: chooses bid strategy based on past win/lose pattern
    
    Args:
        player_id: player ID (0 or 1)
        value: player's value
        round_num: current round number
        history: list of (bid, utility, won) tuples for this player
        env_state: dict with additional info
    
    Returns:
        bid: chosen bid based on feeling (3/4, 1/2, or 1/4 of value)
    """
    if round_num == 0:
        # First round: start with normal strategy
        return value * 0.5
    
    # Analyze past results
    if len(history) == 0:
        return value * 0.5
    
    # Count wins and losses
    wins = sum(1 for _, _, won in history if won)
    losses = sum(1 for _, _, won in history if not won)
    total = len(history)
    
    win_rate = wins / total if total > 0 else 0.5
    
    # Decision rule based on win/lose pattern
    if losses > wins and win_rate < 0.4:
        # Losing many times: use aggressive strategy
        return value * 0.75  # 3/4 of v
    elif wins > losses and win_rate > 0.6:
        # Winning many times: use normal strategy
        return value * 0.5  # 1/2 of v
    else:
        # Mixed win/lose: use conservative strategy
        return value * 0.25  # 1/4 of v

