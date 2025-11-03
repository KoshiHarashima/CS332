# Second, I made algorithm which is tuned to cares about long-term payoff. (2_long Model)
    # This is really difficult to implement. 
    # At each round, it calculate the CDF of other bidders' highest bid.
    # I used time-discounted sum, and algorithm will make long-term payoff.
    # This algorithm represent the smart player in auction.

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