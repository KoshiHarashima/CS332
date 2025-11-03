# First, I made myopic algorithm which is tuned to cares about only current round. (1_myopic Model)
    # This can be implemented by using our Project1's idea!! culculate other bidder's CDF of bid. 
    # At each round, it calculate the CDF of other bidders' highest bid, then calculate the probability of winning.
    # Hence, we get expected value of bid.

# 1_myopic.py
import numpy as np
from typing import List, Tuple

def myopic_algorithm(player_id: int, value: float, round_num: int, 
                     history: List[Tuple[float, float, bool]], 
                     env_state: dict) -> float:
    """
    Myopic algorithm: maximize current round expected utility
    
    Args:
        player_id: player ID (0 or 1)
        value: player's value
        round_num: current round number
        history: list of (bid, utility, won) tuples for this player
        env_state: dict with additional info
    
    Returns:
        bid: optimal bid for current round
    """
    if round_num == 0:
        # First round: no history, bid something conservative
        return value * 0.5
    
    # Extract opponent's bid information from history
    # For each past round:
    #   - If we won: opponent_bid <= our_bid (censored)
    #   - If we lost: opponent_bid = winning_bid (observed)
    
    # Estimate CDF of opponent's bid distribution
    cdf = estimate_opponent_cdf_from_history(history, round_num)
    
    # Calculate expected utility for each possible bid
    # E[u(b)] = (v - b) * P(win|bid=b)
    # where P(win|bid=b) = cdf(b)
    
    # Find bid that maximizes expected utility
    optimal_bid = maximize_expected_utility(value, cdf)
    
    return optimal_bid


def estimate_opponent_cdf_from_history(history: List[Tuple[float, float, bool]], 
                                       round_num: int) -> np.ndarray:
    """
    Estimate CDF of opponent's bid distribution from censored/observed data
    
    Uses Kaplan-Meier estimator or Turnbull estimator for censored data
    """
    # Collect censored and observed data
    observed_bids = []  # When we lost, we know the exact bid
    censored_upper_bounds = []  # When we won, opponent_bid <= our_bid
    
    for bid, utility, won in history:
        if won:
            # We won: opponent's bid <= our bid (right-censored)
            censored_upper_bounds.append(bid)
        else:
            # We lost: opponent's bid = winning bid (observed)
            observed_bids.append(bid)  # This is the bid we lost to
    
    # Use Turnbull estimator or simplified approach:
    # For simplicity, we can use:
    # 1. Observed bids: use empirical CDF
    # 2. Censored data: implies opponent_bid is in [0, upper_bound]
    
    # Create a grid of bid values
    max_bid = max([max(observed_bids, default=0), max(censored_upper_bounds, default=value)])
    bid_grid = np.linspace(0, max_bid, 200)
    
    # Calculate CDF
    cdf = np.zeros_like(bid_grid)
    for i, b in enumerate(bid_grid):
        # Probability that opponent bid <= b
        # = (number of observed bids <= b + number of censored with upper_bound >= b) / total
        count_observed = sum(1 for ob in observed_bids if ob <= b)
        count_censored = sum(1 for ub in censored_upper_bounds if ub >= b)
        total_rounds = len(history)
        cdf[i] = (count_observed + count_censored) / total_rounds
    
    return bid_grid, cdf


def maximize_expected_utility(value: float, cdf: Tuple[np.ndarray, np.ndarray]) -> float:
    """
    Find bid that maximizes expected utility: (v - b) * P(win|bid=b)
    
    Args:
        value: player's value
        cdf: (bid_grid, cdf_values) tuple
    
    Returns:
        optimal_bid: bid that maximizes expected utility
    """
    bid_grid, cdf_values = cdf
    
    # Expected utility = (value - bid) * P(win)
    # P(win) = CDF(bid) if we assume opponent's bid is independent
    expected_utility = (value - bid_grid) * cdf_values
    
    # Find maximum
    optimal_idx = np.argmax(expected_utility)
    optimal_bid = bid_grid[optimal_idx]
    
    # Ensure bid is between 0 and value
    return np.clip(optimal_bid, 0.0, value)