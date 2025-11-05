# repeated_FPA.py
# Repeated First Price Auction simulation and plotting functions for 2 players

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import utility


def play_fpa_round_2players(bid1, bid2):
    """
    Play one round of FPA with 2 players.
    Strictly handles ties with 0.5 probability allocation.
    
    Args:
        bid1: bid from player 1
        bid2: bid from player 2
    
    Returns:
        (allocation1, allocation2): tuple of floats
            - allocation1: allocation for player 1 (1.0 if wins, 0.5 if tie, 0.0 if loses)
            - allocation2: allocation for player 2 (1.0 if wins, 0.5 if tie, 0.0 if loses)
    """
    # Use np.isclose for tie detection to handle floating point precision
    rtol = 1e-9
    atol = 1e-9
    
    if bid1 > bid2 and not np.isclose(bid1, bid2, rtol=rtol, atol=atol):
        return (1.0, 0.0)
    elif bid2 > bid1 and not np.isclose(bid1, bid2, rtol=rtol, atol=atol):
        return (0.0, 1.0)
    else:  # tie (bids are equal within tolerance)
        return (0.5, 0.5)


def run_repeated_fpa(player1_config, player2_config, n_rounds, n_mc, k=100):
    """
    Run repeated First Price Auction simulation
    
    Args:
        player1_config: tuple of (algorithm_function, value, env_state_dict)
        player2_config: tuple of (algorithm_function, value, env_state_dict)
        n_rounds: number of rounds per simulation
        n_mc: number of Monte Carlo runs
        k: number of arms (discretization), default=100
    
    Returns:
        results: dict with regret, utility, win_rate for each player
    """
    alg1_func, v1, env1 = player1_config
    alg2_func, v2, env2 = player2_config
    
    # Initialize environment state for each player
    env1.setdefault('k', k)
    env1.setdefault('h', v1)
    env2.setdefault('k', k)
    env2.setdefault('h', v2)
    
    # Storage for all MC runs
    all_regret1 = []
    all_regret2 = []
    all_utility1 = []
    all_utility2 = []
    all_win_rate1 = []
    all_win_rate2 = []
    all_bids1 = []
    all_bids2 = []
    all_regret_history1 = []
    all_regret_history2 = []
    
    # Monte Carlo loop
    for mc_iter in range(n_mc):
        # Reset for each MC run
        history1 = []
        history2 = []
        env1_state = env1.copy()
        env2_state = env2.copy()
        env1_state['cumulative_payoffs'] = np.zeros(k)
        env2_state['cumulative_payoffs'] = np.zeros(k)
        
        total_utility1 = 0
        total_utility2 = 0
        wins1 = 0
        wins2 = 0
        
        bids1_history = []
        bids2_history = []
        regret_history1 = []
        regret_history2 = []
        
        # Initialize bid grid and cumulative utilities for incremental regret calculation
        bid_grid = np.linspace(0, max(v1, v2), k)
        cumulative_fixed_utility1 = np.zeros(k)  # Cumulative utility for each fixed bid for player 1
        cumulative_fixed_utility2 = np.zeros(k)  # Cumulative utility for each fixed bid for player 2
        best_fixed_bid_idx1 = 0  # Index of best fixed bid for player 1
        best_fixed_bid_idx2 = 0  # Index of best fixed bid for player 2
        
        # Round loop
        for round_num in range(n_rounds):
            # Get bids from each player
            bid1 = alg1_func(0, v1, round_num, history1, env1_state)
            bid2 = alg2_func(1, v2, round_num, history2, env2_state)
            
            bids1_history.append(bid1)
            bids2_history.append(bid2)
            
            # Play FPA
            alloc1, alloc2 = play_fpa_round_2players(bid1, bid2)
            
            # Calculate utilities
            # Utility = allocation * (value - bid)
            utility1 = utility.calculate_utility(v1, alloc1, bid1)
            utility2 = utility.calculate_utility(v2, alloc2, bid2)
            
            # Update totals
            total_utility1 += utility1
            total_utility2 += utility2
            
            # Track wins
            if alloc1 > 0.5:
                wins1 += 1
            elif alloc2 > 0.5:
                wins2 += 1
            else:  # tie
                if np.random.random() < 0.5:
                    wins1 += 1
                else:
                    wins2 += 1
            
            # Update history (Full Feedback: includes opponent's bid)
            won1 = (alloc1 > 0.5) or (alloc1 == 0.5 and np.random.random() < 0.5)
            won2 = (alloc2 > 0.5) or (alloc2 == 0.5 and np.random.random() < 0.5)
            # History format: (bid, utility, won, opponent_bid)
            history1.append((bid1, utility1, won1, bid2))
            history2.append((bid2, utility2, won2, bid1))
            
            # Calculate regret INCREMENTALLY: only compute utility for NEW opponent bid
            # For player 1: add utility from new opponent bid (bid2) to each fixed bid
            for i, fixed_bid in enumerate(bid_grid):
                if fixed_bid <= v1:
                    # Calculate utility for this fixed bid against NEW opponent bid only
                    alloc, _ = play_fpa_round_2players(fixed_bid, bid2)
                    # Utility = allocation * (value - bid)
                    new_utility = utility.calculate_utility(v1, alloc, fixed_bid)
                    cumulative_fixed_utility1[i] += new_utility
            
            # For player 2: add utility from new opponent bid (bid1) to each fixed bid
            for i, fixed_bid in enumerate(bid_grid):
                if fixed_bid <= v2:
                    # Calculate utility for this fixed bid against NEW opponent bid only
                    _, alloc = play_fpa_round_2players(bid1, fixed_bid)
                    # Utility = allocation * (value - bid)
                    new_utility = utility.calculate_utility(v2, alloc, fixed_bid)
                    cumulative_fixed_utility2[i] += new_utility
            
            # Find best fixed bid AFTER updating all fixed bids
            # Only consider bids <= value (valid bids)
            valid_mask1 = bid_grid <= v1
            valid_mask2 = bid_grid <= v2
            
            if np.any(valid_mask1):
                # Find best fixed bid among valid bids for player 1
                valid_utilities1 = cumulative_fixed_utility1.copy()
                valid_utilities1[~valid_mask1] = -np.inf  # Mask invalid bids
                best_fixed_bid_idx1 = np.argmax(valid_utilities1)
            else:
                best_fixed_bid_idx1 = 0
            
            if np.any(valid_mask2):
                # Find best fixed bid among valid bids for player 2
                valid_utilities2 = cumulative_fixed_utility2.copy()
                valid_utilities2[~valid_mask2] = -np.inf  # Mask invalid bids
                best_fixed_bid_idx2 = np.argmax(valid_utilities2)
            else:
                best_fixed_bid_idx2 = 0
            
            # Calculate regret at this round using best fixed bid
            best_fixed_utility1 = cumulative_fixed_utility1[best_fixed_bid_idx1]
            best_fixed_utility2 = cumulative_fixed_utility2[best_fixed_bid_idx2]
            regret1_round = best_fixed_utility1 - total_utility1
            regret2_round = best_fixed_utility2 - total_utility2
            regret_history1.append(regret1_round)
            regret_history2.append(regret2_round)
        
        all_regret1.append(regret_history1[-1])
        all_regret2.append(regret_history2[-1])
        all_utility1.append(total_utility1)
        all_utility2.append(total_utility2)
        all_win_rate1.append(wins1 / n_rounds)
        all_win_rate2.append(wins2 / n_rounds)
        all_bids1.append(bids1_history)
        all_bids2.append(bids2_history)
        all_regret_history1.append(regret_history1)
        all_regret_history2.append(regret_history2)
        
        if (mc_iter + 1) % 10 == 0:
            print(f"MC iteration {mc_iter + 1}/{n_mc} completed")
    
    return {
        'regret1': np.array(all_regret1),
        'regret2': np.array(all_regret2),
        'utility1': np.array(all_utility1),
        'utility2': np.array(all_utility2),
        'win_rate1': np.array(all_win_rate1),
        'win_rate2': np.array(all_win_rate2),
        'bids1': all_bids1,
        'bids2': all_bids2,
        'regret_history1': all_regret_history1,
        'regret_history2': all_regret_history2
    }


def plot_results(results, title="Simulation Results"):
    """
    Plot 4 separate figures and save them as individual PNG files:
    1. Bid evolution over rounds
    2. Regret over time (convergence analysis)
    3. Utility distribution
    4. Win rate distribution
    """
    figures_dir = Path('../figures')
    figures_dir.mkdir(exist_ok=True)
    
    # Create base filename from title
    base_filename = title.lower().replace(' ', '_').replace(' vs ', '_vs_').replace('(', '').replace(')', '')
    
    # Get average bids and regret over MC runs
    bids1_avg = np.mean(results['bids1'], axis=0)
    bids2_avg = np.mean(results['bids2'], axis=0)
    
    # Calculate average regret over time
    regret1_avg = np.mean(results['regret_history1'], axis=0)
    regret2_avg = np.mean(results['regret_history2'], axis=0)
    regret1_std = np.std(results['regret_history1'], axis=0)
    regret2_std = np.std(results['regret_history2'], axis=0)
    
    # Plot 1: Bid evolution over rounds
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    ax1.plot(bids1_avg, label='Player 1', alpha=0.7)
    ax1.plot(bids2_avg, label='Player 2', alpha=0.7)
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Average Bid')
    ax1.set_title('Bid Evolution Over Rounds')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    filepath1 = figures_dir / f"{base_filename}_bid_evolution.png"
    plt.savefig(filepath1, dpi=300, bbox_inches='tight')
    print(f"Plot 1 saved to: {filepath1}")
    plt.close(fig1)
    
    # Plot 2: Regret over time (convergence analysis)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    rounds = np.arange(len(regret1_avg))
    ax2.plot(rounds, regret1_avg, label='Player 1', alpha=0.7)
    ax2.fill_between(rounds, regret1_avg - regret1_std, regret1_avg + regret1_std, alpha=0.2)
    ax2.plot(rounds, regret2_avg, label='Player 2', alpha=0.7)
    ax2.fill_between(rounds, regret2_avg - regret2_std, regret2_avg + regret2_std, alpha=0.2)
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Average Regret')
    ax2.set_title('Regret Over Time (should converge to 0)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.tight_layout()
    filepath2 = figures_dir / f"{base_filename}_regret.png"
    plt.savefig(filepath2, dpi=300, bbox_inches='tight')
    print(f"Plot 2 saved to: {filepath2}")
    plt.close(fig2)
    
    # Plot 4: Utility distribution
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    ax4.hist(results['utility1'], bins=30, alpha=0.5, label='Player 1')
    ax4.hist(results['utility2'], bins=30, alpha=0.5, label='Player 2')
    ax4.set_xlabel('Total Utility')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Total Utility Distribution (MC runs)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    filepath4 = figures_dir / f"{base_filename}_utility_distribution.png"
    plt.savefig(filepath4, dpi=300, bbox_inches='tight')
    print(f"Plot 4 saved to: {filepath4}")
    plt.close(fig4)
    
    # Plot 5: Win rate distribution
    fig5, ax5 = plt.subplots(figsize=(8, 6))
    ax5.hist(results['win_rate1'], bins=30, alpha=0.5, label='Player 1')
    ax5.hist(results['win_rate2'], bins=30, alpha=0.5, label='Player 2')
    ax5.set_xlabel('Win Rate')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Win Rate Distribution (MC runs)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    plt.tight_layout()
    filepath5 = figures_dir / f"{base_filename}_win_rate_distribution.png"
    plt.savefig(filepath5, dpi=300, bbox_inches='tight')
    print(f"Plot 5 saved to: {filepath5}")
    plt.close(fig5)
    
    # Print summary statistics
    print(f"\n=== Summary Statistics ===")
    print(f"Player 1:")
    print(f"  Mean Regret: {np.mean(results['regret1']):.2f} ± {np.std(results['regret1']):.2f}")
    print(f"  Mean Utility: {np.mean(results['utility1']):.2f} ± {np.std(results['utility1']):.2f}")
    print(f"  Mean Win Rate: {np.mean(results['win_rate1']):.3f} ± {np.std(results['win_rate1']):.3f}")
    print(f"\nPlayer 2:")
    print(f"  Mean Regret: {np.mean(results['regret2']):.2f} ± {np.std(results['regret2']):.2f}")
    print(f"  Mean Utility: {np.mean(results['utility2']):.2f} ± {np.std(results['utility2']):.2f}")
    print(f"  Mean Win Rate: {np.mean(results['win_rate2']):.3f} ± {np.std(results['win_rate2']):.3f}")
    
    # Save results to CSV
    save_results_to_csv(results, title)


def save_results_to_csv(results, title="Simulation Results", output_dir="../data"):
    """
    Save simulation results to CSV files
    
    Args:
        results: dict with regret, utility, win_rate for each player
        title: title for the simulation (used for filename)
        output_dir: directory to save CSV files
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create filename from title
    filename = title.lower().replace(' ', '_').replace(' vs ', '_vs_').replace('(', '').replace(')', '')
    
    # Save summary statistics
    summary_data = {
        'metric': ['mean_regret', 'std_regret', 'mean_utility', 'std_utility', 'mean_win_rate', 'std_win_rate'],
        'player1': [
            np.mean(results['regret1']),
            np.std(results['regret1']),
            np.mean(results['utility1']),
            np.std(results['utility1']),
            np.mean(results['win_rate1']),
            np.std(results['win_rate1'])
        ],
        'player2': [
            np.mean(results['regret2']),
            np.std(results['regret2']),
            np.mean(results['utility2']),
            np.std(results['utility2']),
            np.mean(results['win_rate2']),
            np.std(results['win_rate2'])
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_path = output_path / f"{filename}_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary statistics saved to: {summary_path}")
    
    # Save detailed results per MC run
    detailed_data = {
        'mc_run': range(len(results['regret1'])),
        'regret1': results['regret1'],
        'regret2': results['regret2'],
        'utility1': results['utility1'],
        'utility2': results['utility2'],
        'win_rate1': results['win_rate1'],
        'win_rate2': results['win_rate2']
    }
    detailed_df = pd.DataFrame(detailed_data)
    detailed_path = output_path / f"{filename}_detailed.csv"
    detailed_df.to_csv(detailed_path, index=False)
    print(f"Detailed results saved to: {detailed_path}")
    
    # Save regret history over rounds (average across MC runs)
    regret_history_data = {
        'round': np.arange(len(results['regret_history1'][0])),
        'regret1_mean': np.mean(results['regret_history1'], axis=0),
        'regret1_std': np.std(results['regret_history1'], axis=0),
        'regret2_mean': np.mean(results['regret_history2'], axis=0),
        'regret2_std': np.std(results['regret_history2'], axis=0)
    }
    regret_history_df = pd.DataFrame(regret_history_data)
    regret_history_path = output_path / f"{filename}_regret_history.csv"
    regret_history_df.to_csv(regret_history_path, index=False)
    print(f"Regret history saved to: {regret_history_path}")
    
    # Save bid history over rounds (average across MC runs)
    bid_history_data = {
        'round': np.arange(len(results['bids1'][0])),
        'bid1_mean': np.mean(results['bids1'], axis=0),
        'bid1_std': np.std(results['bids1'], axis=0),
        'bid2_mean': np.mean(results['bids2'], axis=0),
        'bid2_std': np.std(results['bids2'], axis=0)
    }
    bid_history_df = pd.DataFrame(bid_history_data)
    bid_history_path = output_path / f"{filename}_bid_history.csv"
    bid_history_df.to_csv(bid_history_path, index=False)
    print(f"Bid history saved to: {bid_history_path}")

