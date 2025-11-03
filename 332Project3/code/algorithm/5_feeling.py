# 5_feeling.py
# Feeling algorithm: uses Exponential Weight to choose among 5 strategies (arms)
# 5 arms:
#   1. E_truth_like: b = v (aggressive)
#   2. E_half: b = 0.5v
#   3. E_quarter: b = 0.25v
#   4. E_win-react: if last=lose then b = min(v, b_prev + 0.1v) else b = b_prev - 0.05v
#   5. E_context_threshold: if last3_wins/3 < 0.33 then b = 0.9v else b = 0.5v
import numpy as np
from typing import List, Tuple

def feeling_algorithm(player_id: int, value: float, round_num: int,
                      history: List[Tuple[float, float, bool]],
                      env_state: dict) -> float:
    """
    Feeling algorithm: Exponential Weight over 5 strategy arms
    
    Args:
        player_id: player ID (0 or 1)
        value: player's value
        round_num: current round number
        history: list of (bid, utility, won) tuples for this player
        env_state: dict with additional info
            - k: number of discrete bids (for discretization), default=100
            - h: scaling parameter, default=value
            - learning_rate: learning rate epsilon, default=1/sqrt(n)
    
    Returns:
        bid: selected bid from discretized grid
    """
    # Get parameters
    k = env_state.get('k', 100)  # discretization
    h = env_state.get('h', value)  # scaling parameter
    learning_rate = env_state.get('learning_rate', None)
    
    # Default learning rate: 1/sqrt(n)
    if learning_rate is None:
        learning_rate = 1.0 / np.sqrt(max(round_num, 1))
    
    # Create bid grid for discretization
    bid_grid = np.linspace(0, value, k)
    
    # Initialize cumulative payoffs for each arm (strategy)
    num_arms = 5
    if 'cumulative_payoffs' not in env_state:
        env_state['cumulative_payoffs'] = np.zeros(num_arms)
    if 'last_selected_arm' not in env_state:
        env_state['last_selected_arm'] = None
    if 'last_bid' not in env_state:
        env_state['last_bid'] = value * 0.5
    
    cumulative_payoffs = env_state['cumulative_payoffs']
    
    # If first round, initialize and return middle bid
    if round_num == 0 or len(history) == 0:
        # Use E_half strategy for first round
        target_bid = value * 0.5
        discrete_bid_idx = np.argmin(np.abs(bid_grid - target_bid))
        env_state['last_selected_arm'] = 1  # E_half
        env_state['last_bid'] = bid_grid[discrete_bid_idx]
        return bid_grid[discrete_bid_idx]
    
    # Update cumulative payoffs from last round's result
    if len(history) > 0 and env_state['last_selected_arm'] is not None:
        last_bid, last_utility, last_won = history[-1]
        arm_idx = env_state['last_selected_arm']
        # Update cumulative payoff for this arm
        cumulative_payoffs[arm_idx] += last_utility
    
    # Define the 5 arms (strategies)
    def arm_bids(value, history, env_state):
        """Calculate bid for each of the 5 arms"""
        bids = np.zeros(num_arms)
        
        # Arm 0: E_truth_like: b = v
        bids[0] = value
        
        # Arm 1: E_half: b = 0.5v
        bids[1] = value * 0.5
        
        # Arm 2: E_quarter: b = 0.25v
        bids[2] = value * 0.25
        
        # Arm 3: E_win-react: if last=lose then b = min(v, b_prev + 0.1v) else b = b_prev - 0.05v
        last_bid_prev = env_state.get('last_bid', value * 0.5)
        if len(history) > 0:
            _, _, last_won = history[-1]
            if not last_won:  # lost
                bids[3] = min(value, last_bid_prev + 0.1 * value)
            else:  # won
                bids[3] = max(0, last_bid_prev - 0.05 * value)
        else:
            bids[3] = value * 0.5
        
        # Arm 4: E_context_threshold: if last3_wins/3 < 0.33 then b = 0.9v else b = 0.5v
        if len(history) >= 3:
            last3_results = [won for _, _, won in history[-3:]]
            last3_wins = sum(last3_results)
            if last3_wins / 3 < 0.33:
                bids[4] = value * 0.9
            else:
                bids[4] = value * 0.5
        else:
            bids[4] = value * 0.5
        
        return bids
    
    # Calculate bids for each arm
    arm_bid_values = arm_bids(value, history, env_state)
    
    # Exponential Weight selection over arms
    # π_j = (1+ε)^(V_j/h) / Σ_j'(1+ε)^(V_j'/h)
    # where V_j = cumulative_payoffs[j]
    if learning_rate == 0:
        # Random selection if learning_rate is 0
        selected_arm = np.random.randint(0, num_arms)
    else:
        # Calculate probabilities using exponential weights
        powers = (1 + learning_rate) ** (cumulative_payoffs / h)
        sum_powers = np.sum(powers)
        
        if not np.isfinite(sum_powers) or sum_powers <= 0:
            # Fallback to random
            selected_arm = np.random.randint(0, num_arms)
        else:
            probabilities = powers / sum_powers
            if not np.all(np.isfinite(probabilities)):
                selected_arm = np.random.randint(0, num_arms)
            else:
                selected_arm = np.random.choice(num_arms, p=probabilities)
    
    # Get bid from selected arm and discretize
    target_bid = arm_bid_values[selected_arm]
    discrete_bid_idx = np.argmin(np.abs(bid_grid - target_bid))
    final_bid = bid_grid[discrete_bid_idx]
    
    # Store for next round
    env_state['last_selected_arm'] = selected_arm
    env_state['last_bid'] = final_bid
    
    return final_bid
