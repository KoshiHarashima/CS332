# bimatrix_wrapper.py
# Wrapper functions to adapt FPA algorithms for 2x2 bimatrix games

import numpy as np
from typing import List, Tuple


def flexible_bimatrix(player_id: int, round_num: int, history: List[Tuple[int, float, int]],
                     env_state: dict, payoff_matrix) -> int:
    """
    Flexible algorithm adapted for 2x2 bimatrix games.
    Uses exponential weights over 2 actions.
    
    Args:
        player_id: player ID (0 or 1)
        round_num: current round number
        history: list of (action, payoff, opponent_action) tuples
        env_state: dict with algorithm state
        payoff_matrix: 2x2x2 payoff matrix
    
    Returns:
        action: 0 or 1
    """
    # Get parameters
    learning_rate = env_state.get('learning_rate', None)
    if learning_rate is None:
        learning_rate = np.sqrt(np.log(2) / max(round_num, 1))
    
    # Initialize cumulative payoffs
    if 'cumulative_payoffs' not in env_state:
        env_state['cumulative_payoffs'] = np.zeros(2)
    
    cumulative_payoffs = env_state['cumulative_payoffs']
    
    # Get player's independent random state from env_state
    random_state = env_state.get('random_state', np.random)
    
    # First round: random action
    if round_num == 0 or len(history) == 0:
        return random_state.randint(0, 2)
    
    # Update cumulative payoffs from last round
    if len(history) > 0:
        last_entry = history[-1]
        my_action, my_payoff, opp_action = last_entry
        
        # For each possible action, calculate what payoff would have been
        for a in range(2):
            potential_payoff = play_bimatrix_round_simple(payoff_matrix, a, opp_action, player_id)
            cumulative_payoffs[a] += potential_payoff
    
    # Exponential weights selection
    if learning_rate == 0:
        return random_state.randint(0, 2)
    
    # Calculate probabilities using exponential weights
    h = env_state.get('h', 1.0)  # scaling parameter
    powers = np.exp(learning_rate * cumulative_payoffs / h)
    sum_powers = np.sum(powers)
    
    if not np.isfinite(sum_powers) or sum_powers <= 0:
        return random_state.randint(0, 2)
    
    probabilities = powers / sum_powers
    if not np.all(np.isfinite(probabilities)):
        return random_state.randint(0, 2)
    
    # Select action
    return random_state.choice(2, p=probabilities)


def play_bimatrix_round_simple(payoff_matrix, action1: int, action2: int, player_id: int):
    """Helper function to get payoff for a player."""
    if player_id == 0:
        return payoff_matrix[action1, action2, 0]
    else:
        return payoff_matrix[action1, action2, 1]


def empirical_bimatrix(player_id: int, round_num: int, history: List[Tuple[int, float, int]],
                      env_state: dict, payoff_matrix) -> int:
    """
    Empirical algorithm adapted for 2x2 bimatrix games.
    Selects action that maximizes expected payoff based on opponent's empirical distribution.
    
    Args:
        player_id: player ID (0 or 1)
        round_num: current round number
        history: list of (action, payoff, opponent_action) tuples
        env_state: dict with algorithm state
        payoff_matrix: 2x2x2 payoff matrix
    
    Returns:
        action: 0 or 1
    """
    # Get player's independent random state from env_state
    random_state = env_state.get('random_state', np.random)
    
    # First round: random action
    if round_num == 0 or len(history) == 0:
        return random_state.randint(0, 2)
    
    # Get opponent's action history
    opponent_actions = [entry[2] for entry in history]
    
    if len(opponent_actions) == 0:
        return random_state.randint(0, 2)
    
    # Calculate empirical distribution of opponent's actions
    opp_dist = np.array([opponent_actions.count(0), opponent_actions.count(1)]) / len(opponent_actions)
    
    # Calculate expected payoff for each action
    expected_payoffs = np.zeros(2)
    for my_action in range(2):
        for opp_action in range(2):
            if player_id == 0:
                payoff = payoff_matrix[my_action, opp_action, 0]
            else:
                payoff = payoff_matrix[my_action, opp_action, 1]
            expected_payoffs[my_action] += opp_dist[opp_action] * payoff
    
    # Select action with highest expected payoff
    return np.argmax(expected_payoffs)


def ftl_bimatrix(player_id: int, round_num: int, history: List[Tuple[int, float, int]],
                 env_state: dict, payoff_matrix) -> int:
    """
    Follow The Leader algorithm adapted for 2x2 bimatrix games.
    Selects action with highest cumulative payoff.
    
    Args:
        player_id: player ID (0 or 1)
        round_num: current round number
        history: list of (action, payoff, opponent_action) tuples
        env_state: dict with algorithm state
        payoff_matrix: 2x2x2 payoff matrix
    
    Returns:
        action: 0 or 1
    """
    # Initialize cumulative payoffs
    if 'cumulative_payoffs' not in env_state:
        env_state['cumulative_payoffs'] = np.zeros(2)
    
    cumulative_payoffs = env_state['cumulative_payoffs']
    
    # Get player's independent random state from env_state
    random_state = env_state.get('random_state', np.random)
    
    # First round: random action
    if round_num == 0 or len(history) == 0:
        return random_state.randint(0, 2)
    
    # Update cumulative payoffs from last round
    if len(history) > 0:
        last_entry = history[-1]
        my_action, my_payoff, opp_action = last_entry
        
        # For each possible action, calculate what payoff would have been
        for a in range(2):
            potential_payoff = play_bimatrix_round_simple(payoff_matrix, a, opp_action, player_id)
            cumulative_payoffs[a] += potential_payoff
    
    # Select action with highest cumulative payoff
    return np.argmax(cumulative_payoffs)

