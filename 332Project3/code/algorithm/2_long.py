# Second, I made algorithm which is tuned to cares about long-term payoff. (2_long Model)
    # This is really difficult to implement. 
    # At each round, it calculate the CDF of other bidders' highest bid.
    # I used time-discounted sum, and algorithm will make long-term payoff.
    # This algorithm represent the smart player in auction.

import numpy as np
from typing import List, Tuple

def long_algorithm(player_id: int, value: float, round_num: int,
                  history: List[Tuple[float, float, bool, float]],
                  env_state: dict) -> float:
    """
    Long-term algorithm: maximizes long-term discounted utility
    
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
        bid: optimal bid for long-term payoff
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
            # Backward compatibility: if old format
            bid, _, won = entry[:3]
            if not won:
                opponent_bids.append(bid)
    
    # Get unified bid grid (k discrete arms in [0, value])
    k = env_state.get('k', 100)
    bid_grid = np.linspace(0, value, k)
    
    # Get discount factor
    delta = env_state.get('delta', 0.95)  # Default discount factor
    remaining_rounds = env_state.get('total_rounds', 1000) - round_num
    
    # Use dynamic programming approach (with strict tie handling)
    optimal_bid = maximize_long_term_utility_dp(value, opponent_bids, bid_grid, delta, remaining_rounds)
    
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
        max_val = np.max(bid_grid)
        if max_val > 0:
            win_probs = np.clip(bid_grid / max_val, 0.0, 1.0)
            return win_probs
        else:
            return np.full_like(bid_grid, 0.5)
    
    # For each bid in grid, calculate P(win)
    # Use relative tolerance for floating point comparison
    rtol = 1e-9
    atol = 1e-9
    
    for i, bid in enumerate(bid_grid):
        # Count observations
        count_less = sum(1 for ob in observed_bids if ob < bid)
        # Use np.isclose with appropriate tolerance for tie detection
        count_equal = sum(1 for ob in observed_bids if np.isclose(ob, bid, rtol=rtol, atol=atol))
        
        # P(win|bid) = P(opponent_bid < bid) + 0.5 * P(opponent_bid == bid)
        p_less = count_less / n_obs
        p_equal = count_equal / n_obs
        win_probs[i] = p_less + 0.5 * p_equal
    
    return win_probs


def maximize_long_term_utility_dp(value: float,
                                 observed_bids: List[float],
                                 bid_grid: np.ndarray,
                                 delta: float,
                                 remaining_rounds: int) -> float:
    """
    Dynamic programming approach for finite horizon with strict tie handling.
    
    Args:
        value: player's value
        observed_bids: list of opponent bids observed so far
        bid_grid: array of bid values to evaluate (must be in [0, value])
        delta: discount factor
        remaining_rounds: number of remaining rounds
    """
    # Calculate win probabilities with strict tie handling
    win_probs = calculate_win_probability(observed_bids, bid_grid)
    
    # Value function V_t(b) = max_bid [ (v-bid)*P(win|bid) + delta*E[V_{t+1}(next_state)] ]
    
    # For simplicity, we approximate by assuming opponent's distribution doesn't change
    # Then the value function becomes stationary: V_t ≈ V_{t+1}
    
    # This simplifies to:
    # V = max_bid [ (v-bid)*P(win|bid) + delta*V ]
    # Solving: V = max_bid [ (v-bid)*P(win|bid) ] / (1 - delta)
    
    one_shot_utility = (value - bid_grid) * win_probs
    max_one_shot = np.max(one_shot_utility)
    
    # Stationary value
    V_stationary = max_one_shot / (1 - delta)
    
    # Current round: choose bid that maximizes current + discounted future
    # U(b) = (v - b)*P(win|b) + delta * V_stationary
    total_utility = (value - bid_grid) * win_probs + delta * V_stationary
    
    # Only consider bids <= value (safety check)
    valid_mask = bid_grid <= value
    if not np.any(valid_mask):
        return 0.0
    
    valid_utility = total_utility.copy()
    valid_utility[~valid_mask] = -np.inf
    optimal_idx = np.argmax(valid_utility)
    
    return np.clip(bid_grid[optimal_idx], 0.0, value)