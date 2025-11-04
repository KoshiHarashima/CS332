# First, I made myopic algorithm which is tuned to cares about only current round. (1_myopic Model)
    # This can be implemented by using our Project1's idea!! culculate other bidder's CDF of bid. 
    # At each round, it calculate the CDF of other bidders' highest bid, then calculate the probability of winning.
    # Hence, we get expected value of bid.

# 1_myopic.py
import numpy as np
from typing import List, Tuple

def myopic_algorithm(player_id: int, value: float, round_num: int, 
                     history: List[Tuple[float, float, bool, float]], 
                     env_state: dict) -> float:
    """
    Myopic algorithm: maximize current round expected utility
    
    Full Information + Full Feedback: Can observe opponent's bids directly.
    
    Args:
        player_id: player ID (0 or 1)
        value: player's value
        round_num: current round number
        history: list of (bid, utility, won, opponent_bid) tuples for this player
                 - bid: player's bid
                 - utility: player's utility
                 - won: whether player won
                 - opponent_bid: opponent's bid (Full Feedback)
        env_state: dict with additional info
    
    Returns:
        bid: optimal bid for current round
    """
    if round_num == 0:
        # First round: no history, bid something conservative (discretized)
        k = env_state.get('k', 100)
        bid_grid = np.linspace(0, value, k)
        target_bid = value * 0.5
        discrete_bid_idx = np.argmin(np.abs(bid_grid - target_bid))
        return bid_grid[discrete_bid_idx]
    
    # Full Feedback: Extract opponent's bids directly from history
    # History format: (bid, utility, won, opponent_bid)
    opponent_bids = []
    for entry in history:
        if len(entry) >= 4:
            _, _, _, opp_bid = entry
            opponent_bids.append(opp_bid)
        else:
            # Backward compatibility: if old format, estimate from won/lost
            bid, _, won = entry[:3]
            if not won:
                opponent_bids.append(bid)  # Lost, so opponent bid >= our bid
    
    # Get unified bid grid (k discrete arms in [0, value])
    k = env_state.get('k', 100)
    bid_grid = np.linspace(0, value, k)
    
    # Calculate expected utility for each possible bid
    # E[u(b)] = (v - b) * P(win|bid=b)
    # where P(win|bid=b) = P(opponent_bid < b) + 0.5 * P(opponent_bid == b)
    
    # Find bid that maximizes expected utility (on unified grid)
    optimal_bid = maximize_expected_utility(value, opponent_bids, bid_grid)
    
    # Optimal bid is already on the grid, but ensure we return exact grid value
    discrete_bid_idx = np.argmin(np.abs(bid_grid - optimal_bid))
    return bid_grid[discrete_bid_idx]


def calculate_win_probability(observed_bids: List[float], bid_grid: np.ndarray) -> np.ndarray:
    """
    Calculate P(win|bid=b) for each bid in bid_grid, strictly handling ties.
    
    In First Price Auction with tie-breaking:
    - P(win|bid=b) = P(opponent_bid < b) + 0.5 * P(opponent_bid == b)
    
    Args:
        observed_bids: list of opponent bids observed so far
        bid_grid: array of bid values to evaluate
    
    Returns:
        win_probs: array of win probabilities for each bid in bid_grid
    """
    n_obs = len(observed_bids)
    win_probs = np.zeros_like(bid_grid)
    
    if n_obs == 0:
        # No observations: assume uniform distribution
        # P(win) = bid / value (assuming opponent bids uniformly in [0, value])
        # Use bid_grid to infer value (max value in grid)
        max_val = np.max(bid_grid)
        if max_val > 0:
            # P(win|bid) ≈ bid / max_val for uniform distribution
            # But ensure it's reasonable: if bid is close to max_val, P(win) should be close to 1
            win_probs = np.clip(bid_grid / max_val, 0.0, 1.0)
            return win_probs
        else:
            return np.full_like(bid_grid, 0.5)
    
    # Vectorized calculation: compute P(win) for all bids at once
    # Use relative tolerance for floating point comparison
    rtol = 1e-9
    atol = 1e-9
    
    # Convert observed_bids to numpy array for vectorized operations
    observed_bids_arr = np.array(observed_bids)
    
    # For each bid in grid, calculate P(win) using vectorized operations
    # Shape: (len(bid_grid), len(observed_bids))
    bid_grid_2d = bid_grid[:, np.newaxis]  # Shape: (len(bid_grid), 1)
    observed_bids_2d = observed_bids_arr[np.newaxis, :]  # Shape: (1, len(observed_bids))
    
    # Count how many observed bids are less than each bid (vectorized)
    count_less = np.sum(observed_bids_2d < bid_grid_2d, axis=1)  # Shape: (len(bid_grid),)
    
    # Count how many observed bids are equal to each bid (vectorized, using np.isclose)
    count_equal = np.sum(
        np.isclose(observed_bids_2d, bid_grid_2d, rtol=rtol, atol=atol),
        axis=1
    )  # Shape: (len(bid_grid),)
    
    # P(win|bid) = P(opponent_bid < bid) + 0.5 * P(opponent_bid == bid)
    win_probs = (count_less + 0.5 * count_equal) / n_obs
    
    return win_probs


def maximize_expected_utility(value: float, observed_bids: List[float], 
                              bid_grid: np.ndarray) -> float:
    """
    Find bid that maximizes expected utility: (v - b) * P(win|bid=b)
    
    Strictly handles ties with 0.5 probability allocation.
    
    Args:
        value: player's value
        observed_bids: list of opponent bids observed so far
        bid_grid: array of bid values to evaluate (must be in [0, value])
    
    Returns:
        optimal_bid: bid that maximizes expected utility
    """
    # Calculate win probabilities for each bid
    win_probs = calculate_win_probability(observed_bids, bid_grid)
    
    # Expected utility = (value - bid) * P(win|bid)
    expected_utility = (value - bid_grid) * win_probs
    
    # Only consider bids <= value (safety check)
    valid_mask = bid_grid <= value
    if not np.any(valid_mask):
        return 0.0
    
    # Find maximum among valid bids
    valid_utility = expected_utility.copy()
    valid_utility[~valid_mask] = -np.inf
    optimal_idx = np.argmax(valid_utility)
    optimal_bid = bid_grid[optimal_idx]
    
    # Ensure bid is between 0 and value
    return np.clip(optimal_bid, 0.0, value)