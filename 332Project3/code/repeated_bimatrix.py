# repeated_bimatrix.py
# Repeated 2x2 Bimatrix Game simulation for 2 players
# Focus on coordination games with multiple Nash equilibria

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from typing import List, Tuple, Callable


def play_bimatrix_round_simple(payoff_matrix, action1: int, action2: int, player_id: int):
    """Helper function to get payoff for a player."""
    if player_id == 0:
        return payoff_matrix[action1, action2, 0]
    else:
        return payoff_matrix[action1, action2, 1]


def play_bimatrix_round(payoff_matrix, action1: int, action2: int):
    """
    Play one round of 2x2 bimatrix game.
    
    Args:
        payoff_matrix: 2x2x2 array where payoff_matrix[i, j, 0] is player 1's payoff
                      and payoff_matrix[i, j, 1] is player 2's payoff when
                      player 1 plays action i and player 2 plays action j
        action1: player 1's action (0 or 1)
        action2: player 2's action (0 or 1)
    
    Returns:
        (payoff1, payoff2): tuple of payoffs
    """
    return (payoff_matrix[action1, action2, 0], payoff_matrix[action1, action2, 1])


def run_repeated_bimatrix(player1_config, player2_config, payoff_matrix, n_rounds, n_mc):
    """
    Run repeated 2x2 bimatrix game simulation
    
    Args:
        player1_config: tuple of (algorithm_function, env_state_dict)
        player2_config: tuple of (algorithm_function, env_state_dict)
        payoff_matrix: 2x2x2 array of payoffs
        n_rounds: number of rounds per simulation
        n_mc: number of Monte Carlo runs
    
    Returns:
        results: dict with action history, payoff history, NE convergence info
    """
    alg1_func, env1 = player1_config
    alg2_func, env2 = player2_config
    
    # Storage for all MC runs
    all_actions1 = []
    all_actions2 = []
    all_payoffs1 = []
    all_payoffs2 = []
    all_action_history1 = []
    all_action_history2 = []
    all_payoff_history1 = []
    all_payoff_history2 = []
    
    # Find Nash equilibria
    ne_pure = find_pure_nash_equilibria(payoff_matrix)
    ne_mixed = find_mixed_nash_equilibria(payoff_matrix)
    
    # Monte Carlo loop
    for mc_iter in range(n_mc):
        # Reset for each MC run
        history1 = []
        history2 = []
        env1_state = env1.copy()
        env2_state = env2.copy()
        
        # Initialize cumulative payoffs for each action
        if 'cumulative_payoffs' not in env1_state:
            env1_state['cumulative_payoffs'] = np.zeros(2)
        if 'cumulative_payoffs' not in env2_state:
            env2_state['cumulative_payoffs'] = np.zeros(2)
        
        total_payoff1 = 0
        total_payoff2 = 0
        
        # Pre-allocate arrays
        actions1_history = np.zeros(n_rounds, dtype=int)
        actions2_history = np.zeros(n_rounds, dtype=int)
        payoffs1_history = np.zeros(n_rounds)
        payoffs2_history = np.zeros(n_rounds)
        
        # Round loop
        for round_num in range(n_rounds):
            # Get actions from each player
            # For bimatrix games, algorithms should return action index (0 or 1)
            action1 = alg1_func(0, round_num, history1, env1_state, payoff_matrix)
            action2 = alg2_func(1, round_num, history2, env2_state, payoff_matrix)
            
            actions1_history[round_num] = action1
            actions2_history[round_num] = action2
            
            # Play game
            payoff1, payoff2 = play_bimatrix_round(payoff_matrix, action1, action2)
            
            payoffs1_history[round_num] = payoff1
            payoffs2_history[round_num] = payoff2
            
            # Update totals
            total_payoff1 += payoff1
            total_payoff2 += payoff2
            
            # History format: (action, payoff, opponent_action)
            history1.append((action1, payoff1, action2))
            history2.append((action2, payoff2, action1))
            
            # Update cumulative payoffs for regret calculation
            # For each action, calculate what payoff would have been
            for a1 in range(2):
                potential_payoff1, _ = play_bimatrix_round(payoff_matrix, a1, action2)
                env1_state['cumulative_payoffs'][a1] += potential_payoff1
            
            for a2 in range(2):
                _, potential_payoff2 = play_bimatrix_round(payoff_matrix, action1, a2)
                env2_state['cumulative_payoffs'][a2] += potential_payoff2
        
        all_actions1.append(actions1_history)
        all_actions2.append(actions2_history)
        all_payoffs1.append(total_payoff1)
        all_payoffs2.append(total_payoff2)
        all_action_history1.append(actions1_history)
        all_action_history2.append(actions2_history)
        all_payoff_history1.append(payoffs1_history)
        all_payoff_history2.append(payoffs2_history)
        
        if (mc_iter + 1) % 10 == 0:
            print(f"MC iteration {mc_iter + 1}/{n_mc} completed")
    
    return {
        'actions1': all_actions1,
        'actions2': all_actions2,
        'payoffs1': np.array(all_payoffs1),
        'payoffs2': np.array(all_payoffs2),
        'action_history1': all_action_history1,
        'action_history2': all_action_history2,
        'payoff_history1': all_payoff_history1,
        'payoff_history2': all_payoff_history2,
        'ne_pure': ne_pure,
        'ne_mixed': ne_mixed,
        'payoff_matrix': payoff_matrix
    }


def find_pure_nash_equilibria(payoff_matrix):
    """
    Find all pure strategy Nash equilibria in a 2x2 game.
    
    Returns:
        list of tuples (action1, action2) representing pure NE
    """
    ne_list = []
    for a1 in range(2):
        for a2 in range(2):
            # Check if (a1, a2) is a NE
            # Player 1: check if a1 is best response to a2
            payoff1_current = payoff_matrix[a1, a2, 0]
            payoff1_other = payoff_matrix[1 - a1, a2, 0]
            
            # Player 2: check if a2 is best response to a1
            payoff2_current = payoff_matrix[a1, a2, 1]
            payoff2_other = payoff_matrix[a1, 1 - a2, 1]
            
            if payoff1_current >= payoff1_other and payoff2_current >= payoff2_other:
                ne_list.append((a1, a2))
    
    return ne_list


def find_mixed_nash_equilibria(payoff_matrix):
    """
    Find mixed strategy Nash equilibria in a 2x2 game.
    For simplicity, we'll return None and focus on pure NE.
    """
    # Mixed NE calculation is more complex, skip for now
    return None


def plot_ne_convergence(results, title="NE Convergence Analysis", save_dir=None):
    """
    Plot action convergence and NE selection.
    
    Args:
        results: dict with simulation results
        title: title for the plots
        save_dir: directory to save plots
    """
    if save_dir is None:
        save_dir = Path('../figures')
    save_dir.mkdir(exist_ok=True)
    
    base_filename = title.lower().replace(' ', '_').replace('(', '').replace(')', '')
    n_mc = len(results['actions1'])
    n_rounds = len(results['actions1'][0])
    
    # Calculate action frequencies over time
    # For each round, calculate fraction of MC runs where each action profile occurred
    action_profile_freq = np.zeros((n_rounds, 4))  # (A,A), (A,B), (B,A), (B,B)
    
    for round_num in range(n_rounds):
        for mc_run in range(n_mc):
            a1 = results['actions1'][mc_run][round_num]
            a2 = results['actions2'][mc_run][round_num]
            profile_idx = a1 * 2 + a2
            action_profile_freq[round_num, profile_idx] += 1
    
    action_profile_freq = action_profile_freq / n_mc
    
    # Plot action profile frequencies over time
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Action profile frequencies
    ax1 = axes[0]
    rounds = np.arange(n_rounds)
    ax1.plot(rounds, action_profile_freq[:, 0], label='(A, A)', linewidth=2, alpha=0.8)
    ax1.plot(rounds, action_profile_freq[:, 1], label='(A, B)', linewidth=2, alpha=0.8)
    ax1.plot(rounds, action_profile_freq[:, 2], label='(B, A)', linewidth=2, alpha=0.8)
    ax1.plot(rounds, action_profile_freq[:, 3], label='(B, B)', linewidth=2, alpha=0.8)
    
    # Mark Nash equilibria
    ne_pure = results['ne_pure']
    for ne in ne_pure:
        a1, a2 = ne
        profile_idx = a1 * 2 + a2
        profile_name = f"({chr(65+a1)}, {chr(65+a2)})"
        ax1.axhline(y=1.0 if profile_idx == 0 else 0.0, color='red', 
                   linestyle='--', alpha=0.5, linewidth=1)
        ax1.text(n_rounds * 0.95, action_profile_freq[-1, profile_idx] + 0.05,
                f'NE: {profile_name}', fontsize=10, color='red', weight='bold')
    
    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('Action Profile Frequency', fontsize=12)
    ax1.set_title('Action Profile Frequencies Over Time', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-0.05, 1.05])
    
    # Plot 2: Final NE convergence (last 10% of rounds)
    ax2 = axes[1]
    last_n_rounds = max(100, n_rounds // 10)
    final_rounds = action_profile_freq[-last_n_rounds:, :]
    final_freq = np.mean(final_rounds, axis=0)
    final_std = np.std(final_rounds, axis=0)
    
    x_pos = np.arange(4)
    profile_names = ['(A, A)', '(A, B)', '(B, A)', '(B, B)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    bars = ax2.bar(x_pos, final_freq, yerr=final_std, capsize=5, alpha=0.7,
                   color=colors, edgecolor='black', linewidth=1.5)
    
    # Highlight NE
    for i, ne in enumerate(ne_pure):
        a1, a2 = ne
        profile_idx = a1 * 2 + a2
        bars[profile_idx].set_edgecolor('red')
        bars[profile_idx].set_linewidth(3)
    
    ax2.set_ylabel('Average Frequency (Last 10% Rounds)', fontsize=12)
    ax2.set_xlabel('Action Profile', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(profile_names)
    ax2.set_title('Final NE Convergence', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 1.1])
    
    # Add value labels
    for bar, val, err in zip(bars, final_freq, final_std):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + err + 0.02,
                f'{val:.2f}±{err:.2f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    filepath = save_dir / f"{base_filename}_ne_convergence.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"NE convergence plot saved to: {filepath}")
    plt.close(fig)
    
    # Print summary
    print(f"\n=== NE Convergence Summary ===")
    print(f"Pure Nash Equilibria: {ne_pure}")
    print(f"Final action profile frequencies (last {last_n_rounds} rounds):")
    for i, name in enumerate(profile_names):
        is_ne = (i // 2, i % 2) in ne_pure
        marker = " [NE]" if is_ne else ""
        print(f"  {name}: {final_freq[i]:.3f} ± {final_std[i]:.3f}{marker}")

