# Second, I made algorithm which is tuned to cares about long-term payoff. (2_long Model)
    # This is really difficult to implement. 
    # At each round, it calculate the CDF of other bidders' highest bid.
    # I used time-discounted sum, and algorithm will make long-term payoff.
    # This algorithm represent the smart player in auction.

import numpy as np
from typing import List, Tuple

def long_algorithm(player_id: int, value: float, round_num: int,
                  history: List[Tuple[float, float, bool]],
                  env_state: dict) -> float:
    """
    Long-term algorithm: maximizes long-term discounted utility
    
    Args:
        player_id: player ID (0 or 1)
        value: player's value
        round_num: current round number
        history: list of (bid, utility, won) tuples for this player
        env_state: dict with additional info
    
    Returns:
        bid: optimal bid for long-term payoff
    """
    if round_num == 0:
        # First round: no history, bid something conservative (discretized)
        k = env_state.get('k', 100)
        bid_grid = np.linspace(0, value, k)
        target_bid = value * 0.5
        discrete_bid_idx = np.argmin(np.abs(bid_grid - target_bid))
        return bid_grid[discrete_bid_idx]
    
    # Estimate opponent's bid distribution (similar to myopic)
    # Extract opponent's bid information from history
    observed_bids = []
    censored_upper_bounds = []
    
    for bid, utility, won in history:
        if won:
            censored_upper_bounds.append(bid)
        else:
            observed_bids.append(bid)
    
    # Create bid grid
    max_observed = max(observed_bids) if observed_bids else 0
    max_censored = max(censored_upper_bounds) if censored_upper_bounds else value
    max_bid = max(max_observed, max_censored, value)
    bid_grid = np.linspace(0, max_bid, 200)
    
    # Calculate CDF
    cdf_values = np.zeros_like(bid_grid)
    for i, b in enumerate(bid_grid):
        count_observed = sum(1 for ob in observed_bids if ob <= b)
        count_censored = sum(1 for ub in censored_upper_bounds if ub >= b)
        total_rounds = len(history)
        cdf_values[i] = (count_observed + count_censored) / total_rounds if total_rounds > 0 else 0
    
    opponent_cdf = (bid_grid, cdf_values)
    
    # Get discount factor
    delta = env_state.get('delta', 0.95)  # Default discount factor
    remaining_rounds = env_state.get('total_rounds', 1000) - round_num
    
    # Use dynamic programming approach
    optimal_bid = maximize_long_term_utility_dp(value, opponent_cdf, delta, remaining_rounds)
    
    # Discretize: select from k discrete arms
    k = env_state.get('k', 100)
    bid_grid = np.linspace(0, value, k)
    discrete_bid_idx = np.argmin(np.abs(bid_grid - optimal_bid))
    return bid_grid[discrete_bid_idx]


def maximize_long_term_utility_dp(value: float,
                                 opponent_cdf: Tuple[np.ndarray, np.ndarray],
                                 delta: float,
                                 remaining_rounds: int) -> float:
    """
    Dynamic programming approach for finite horizon
    """
    bid_grid, cdf_values = opponent_cdf
    
    # Value function V_t(b) = max_bid [ (v-bid)*P(win|bid) + delta*E[V_{t+1}(next_state)] ]
    
    # For simplicity, we approximate by assuming opponent's distribution doesn't change
    # Then the value function becomes stationary: V_t ≈ V_{t+1}
    
    # This simplifies to:
    # V = max_bid [ (v-bid)*P(win|bid) + delta*V ]
    # Solving: V = max_bid [ (v-bid)*P(win|bid) ] / (1 - delta)
    
    one_shot_utility = (value - bid_grid) * cdf_values
    max_one_shot = np.max(one_shot_utility)
    
    # Stationary value
    V_stationary = max_one_shot / (1 - delta)
    
    # Current round: choose bid that maximizes current + discounted future
    # U(b) = (v - b)*P(win|b) + delta * V_stationary
    total_utility = (value - bid_grid) * cdf_values + delta * V_stationary
    
    optimal_idx = np.argmax(total_utility)
    return np.clip(bid_grid[optimal_idx], 0.0, value)