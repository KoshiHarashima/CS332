# n_repeatedFPA.py
# Repeated First Price Auction simulation and plotting functions for n players

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import utility


def play_fpa_round(bids):
    """
    Play one round of FPA with n players.
    Strictly handles ties with equal probability allocation among winners.
    
    Args:
        bids: list of bids from n players [bid1, bid2, ..., bidn]
    
    Returns:
        allocations: list of allocations [alloc1, alloc2, ..., allocn]
            - 1.0 if player wins alone
            - 0.0 if player loses
            - 1.0/num_winners if there's a tie (shared equally among winners)
    """
    n_players = len(bids)
    max_bid = max(bids)
    
    # Use np.isclose for tie detection to handle floating point precision
    rtol = 1e-9
    atol = 1e-9
    winners = [i for i, bid in enumerate(bids) 
               if np.isclose(bid, max_bid, rtol=rtol, atol=atol)]
    n_winners = len(winners)
    
    allocations = [0.0] * n_players
    if n_winners == 1:
        # Single winner
        allocations[winners[0]] = 1.0
    else:
        # Tie: allocate equally among winners (each gets 1.0/n_winners)
        allocation_per_winner = 1.0 / n_winners
        for winner_idx in winners:
            allocations[winner_idx] = allocation_per_winner
    
    return allocations


def run_repeated_fpa_n_players(player_configs, n_rounds, n_mc, k=100):
    """
    Run repeated First Price Auction simulation for n players
    
    Args:
        player_configs: list of tuples, each tuple is (algorithm_function, value, env_state_dict)
            Example: [
                (alg_func1, v1, env1),
                (alg_func2, v2, env2),
                ...
                (alg_funcn, vn, envn)
            ]
        n_rounds: number of rounds per simulation
        n_mc: number of Monte Carlo runs
        k: number of arms (discretization), default=100
    
    Returns:
        results: dict with regret, utility, win_rate for each player
            - 'regret{i}': final regret for player i
            - 'utility{i}': total utility for player i
            - 'win_rate{i}': win rate for player i
            - 'bids{i}': bid history for player i
            - 'regret_history{i}': regret history over rounds for player i
    """
    n_players = len(player_configs)
    
    # Initialize environment state for each player
    for i, (alg_func, value, env) in enumerate(player_configs):
        env.setdefault('k', k)
        env.setdefault('h', value)
    
    # Storage for all MC runs
    all_regret = {f'regret{i}': [] for i in range(n_players)}
    all_utility = {f'utility{i}': [] for i in range(n_players)}
    all_win_rate = {f'win_rate{i}': [] for i in range(n_players)}
    all_bids = {f'bids{i}': [] for i in range(n_players)}
    all_regret_history = {f'regret_history{i}': [] for i in range(n_players)}
    
    # Monte Carlo loop
    for mc_iter in range(n_mc):
        # Reset for each MC run
        histories = [[] for _ in range(n_players)]
        env_states = [env.copy() for _, _, env in player_configs]
        for env_state in env_states:
            env_state['cumulative_payoffs'] = np.zeros(k)
        
        total_utilities = [0.0] * n_players
        wins = [0] * n_players
        
        bids_histories = [[] for _ in range(n_players)]
        regret_histories = [[] for _ in range(n_players)]
        opponent_bids = [[] for _ in range(n_players)]  # opponent_bids[i] stores bids from all other players
        
        # Round loop
        for round_num in range(n_rounds):
            # Get bids from each player
            bids = []
            for i, (alg_func, value, _) in enumerate(player_configs):
                bid = alg_func(i, value, round_num, histories[i], env_states[i])
                bids.append(bid)
                bids_histories[i].append(bid)
            
            # Play FPA
            allocations = play_fpa_round(bids)
            
            # Calculate utilities
            utilities = []
            for i, (_, value, _) in enumerate(player_configs):
                util = utility.calculate_utility(value, allocations[i], bids[i])
                utilities.append(util)
                total_utilities[i] += util
            
            # Track wins (player wins if allocation > 0.5, or wins tie with probability)
            for i in range(n_players):
                if allocations[i] > 0.5:
                    wins[i] += 1
                elif allocations[i] > 0.0 and allocations[i] <= 0.5:
                    # Tie: randomly decide winner
                    if np.random.random() < allocations[i]:
                        wins[i] += 1
            
            # Update history (Full Feedback: includes all opponents' bids)
            for i in range(n_players):
                won = (allocations[i] > 0.5) or (allocations[i] > 0.0 and np.random.random() < allocations[i])
                # History format: (bid, utility, won, opponent_bids_list)
                # For n players, opponent_bids_list contains bids from all other players
                opponent_bids_for_i = [bids[j] for j in range(n_players) if j != i]
                histories[i].append((bids[i], utilities[i], won, opponent_bids_for_i))
            
            # Store opponent bids for regret calculation (Full Feedback)
            for i in range(n_players):
                opponent_bids[i].append([bids[j] for j in range(n_players) if j != i])
            
            # Calculate regret: best fixed action in hindsight
            max_value = max(v for _, v, _ in player_configs)
            bid_grid = np.linspace(0, max_value, k)
            
            for i, (_, value, _) in enumerate(player_configs):
                # Find best fixed bid in hindsight using actual opponent bids
                best_fixed_utility = 0
                for fixed_bid in bid_grid:
                    if fixed_bid > value:
                        continue
                    
                    # Calculate utility if player i always bid fixed_bid
                    fixed_total_utility = 0
                    for round_idx in range(round_num + 1):
                        # Get actual bids from other players in this round
                        actual_bids_round = [bids_histories[j][round_idx] for j in range(n_players)]
                        # Replace player i's bid with fixed_bid
                        test_bids = actual_bids_round.copy()
                        test_bids[i] = fixed_bid
                        # Calculate allocation
                        test_allocations = play_fpa_round(test_bids)
                        # Calculate utility
                        test_util = utility.calculate_utility(value, test_allocations[i], fixed_bid)
                        fixed_total_utility += test_util
                    
                    best_fixed_utility = max(best_fixed_utility, fixed_total_utility)
                
                # Calculate regret at this round
                regret_i_round = best_fixed_utility - total_utilities[i]
                regret_histories[i].append(regret_i_round)
        
        # Store results for this MC run
        for i in range(n_players):
            all_regret[f'regret{i}'].append(regret_histories[i][-1])
            all_utility[f'utility{i}'].append(total_utilities[i])
            all_win_rate[f'win_rate{i}'].append(wins[i] / n_rounds)
            all_bids[f'bids{i}'].append(bids_histories[i])
            all_regret_history[f'regret_history{i}'].append(regret_histories[i])
        
        if (mc_iter + 1) % 10 == 0:
            print(f"MC iteration {mc_iter + 1}/{n_mc} completed")
    
    # Convert to numpy arrays
    results = {}
    for i in range(n_players):
        results[f'regret{i}'] = np.array(all_regret[f'regret{i}'])
        results[f'utility{i}'] = np.array(all_utility[f'utility{i}'])
        results[f'win_rate{i}'] = np.array(all_win_rate[f'win_rate{i}'])
        results[f'bids{i}'] = all_bids[f'bids{i}']
        results[f'regret_history{i}'] = all_regret_history[f'regret_history{i}']
    
    return results


def plot_n_players_results(results, title="Simulation Results", player_labels=None):
    """
    Plot regret, utility, and win rate over rounds for n players
    
    Args:
        results: dict with results from run_repeated_fpa_n_players
        title: plot title
        player_labels: list of labels for players (optional)
    """
    n_players = len([k for k in results.keys() if k.startswith('regret')])
    
    if player_labels is None:
        player_labels = [f'Player {i+1}' for i in range(n_players)]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Get average bids and regret over MC runs
    bids_avg = {}
    regret_avg = {}
    regret_std = {}
    
    for i in range(n_players):
        bids_avg[i] = np.mean(results[f'bids{i}'], axis=0)
        regret_avg[i] = np.mean(results[f'regret_history{i}'], axis=0)
        regret_std[i] = np.std(results[f'regret_history{i}'], axis=0)
    
    # Plot 1: Bid evolution over rounds
    for i in range(n_players):
        axes[0, 0].plot(bids_avg[i], label=player_labels[i], alpha=0.7)
    axes[0, 0].set_xlabel('Round')
    axes[0, 0].set_ylabel('Average Bid')
    axes[0, 0].set_title('Bid Evolution Over Rounds')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Regret over time (convergence analysis)
    rounds = np.arange(len(regret_avg[0]))
    for i in range(n_players):
        axes[0, 1].plot(rounds, regret_avg[i], label=player_labels[i], alpha=0.7)
        axes[0, 1].fill_between(rounds, regret_avg[i] - regret_std[i], 
                                regret_avg[i] + regret_std[i], alpha=0.2)
    axes[0, 1].set_xlabel('Round')
    axes[0, 1].set_ylabel('Average Regret')
    axes[0, 1].set_title('Regret Over Time (should converge to 0)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Plot 3: Regret distribution (final)
    for i in range(n_players):
        axes[0, 2].hist(results[f'regret{i}'], bins=30, alpha=0.5, label=player_labels[i])
    axes[0, 2].set_xlabel('Final Regret')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Final Regret Distribution (MC runs)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Utility distribution
    for i in range(n_players):
        axes[1, 0].hist(results[f'utility{i}'], bins=30, alpha=0.5, label=player_labels[i])
    axes[1, 0].set_xlabel('Total Utility')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Total Utility Distribution (MC runs)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Win rate distribution
    for i in range(n_players):
        axes[1, 1].hist(results[f'win_rate{i}'], bins=30, alpha=0.5, label=player_labels[i])
    axes[1, 1].set_xlabel('Win Rate')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Win Rate Distribution (MC runs)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Bid variance over time (stability measure)
    for i in range(n_players):
        bid_variance = np.var(results[f'bids{i}'], axis=0)
        axes[1, 2].plot(bid_variance, label=player_labels[i], alpha=0.7)
    axes[1, 2].set_xlabel('Round')
    axes[1, 2].set_ylabel('Bid Variance')
    axes[1, 2].set_title('Bid Stability Over Time')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    # Save figure to figures directory
    figures_dir = Path('../figures')
    figures_dir.mkdir(exist_ok=True)
    
    # Create short filename from title
    filename = title.lower().replace(' ', '_').replace(' vs ', '_vs_')
    filepath = figures_dir / f"{filename}.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {filepath}")
    
    plt.show()
    
    # Print summary statistics
    print(f"\n=== Summary Statistics ===")
    for i in range(n_players):
        print(f"{player_labels[i]}:")
        print(f"  Mean Regret: {np.mean(results[f'regret{i}']):.2f} ± {np.std(results[f'regret{i}']):.2f}")
        print(f"  Mean Utility: {np.mean(results[f'utility{i}']):.2f} ± {np.std(results[f'utility{i}']):.2f}")
        print(f"  Mean Win Rate: {np.mean(results[f'win_rate{i}']):.3f} ± {np.std(results[f'win_rate{i}']):.3f}")

