# two_repeatedFPA.py
# Repeated First Price Auction simulation and plotting functions for 2 players

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
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
        opponent_bids1 = []
        opponent_bids2 = []
        
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
            
            # Store opponent bids for regret calculation (Full Feedback)
            opponent_bids1.append(bid2)
            opponent_bids2.append(bid1)
            
            # Calculate regret: best fixed action in hindsight
            bid_grid = np.linspace(0, max(v1, v2), k)
            
            best_fixed_utility1 = 0
            for fixed_bid in bid_grid:
                if fixed_bid > v1:
                    continue
                fixed_utility1 = sum([
                    utility.calculate_utility(v1,
                        play_fpa_round_2players(fixed_bid, opp_bid)[0],
                        fixed_bid)
                    for opp_bid in opponent_bids1
                ])
                best_fixed_utility1 = max(best_fixed_utility1, fixed_utility1)
            
            best_fixed_utility2 = 0
            for fixed_bid in bid_grid:
                if fixed_bid > v2:
                    continue
                fixed_utility2 = sum([
                    utility.calculate_utility(v2,
                        play_fpa_round_2players(opp_bid, fixed_bid)[1],
                        fixed_bid)
                    for opp_bid in opponent_bids2
                ])
                best_fixed_utility2 = max(best_fixed_utility2, fixed_utility2)
            
            # Calculate regret at this round
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


def plot_part1_results(results, title="Simulation Results"):
    """
    Plot regret, utility, and win rate over rounds
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Get average bids and regret over MC runs
    bids1_avg = np.mean(results['bids1'], axis=0)
    bids2_avg = np.mean(results['bids2'], axis=0)
    
    # Calculate average regret over time
    regret1_avg = np.mean(results['regret_history1'], axis=0)
    regret2_avg = np.mean(results['regret_history2'], axis=0)
    regret1_std = np.std(results['regret_history1'], axis=0)
    regret2_std = np.std(results['regret_history2'], axis=0)
    
    # Plot 1: Bid evolution over rounds
    axes[0, 0].plot(bids1_avg, label='Player 1', alpha=0.7)
    axes[0, 0].plot(bids2_avg, label='Player 2', alpha=0.7)
    axes[0, 0].set_xlabel('Round')
    axes[0, 0].set_ylabel('Average Bid')
    axes[0, 0].set_title('Bid Evolution Over Rounds')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Regret over time (convergence analysis)
    rounds = np.arange(len(regret1_avg))
    axes[0, 1].plot(rounds, regret1_avg, label='Player 1', alpha=0.7)
    axes[0, 1].fill_between(rounds, regret1_avg - regret1_std, regret1_avg + regret1_std, alpha=0.2)
    axes[0, 1].plot(rounds, regret2_avg, label='Player 2', alpha=0.7)
    axes[0, 1].fill_between(rounds, regret2_avg - regret2_std, regret2_avg + regret2_std, alpha=0.2)
    axes[0, 1].set_xlabel('Round')
    axes[0, 1].set_ylabel('Average Regret')
    axes[0, 1].set_title('Regret Over Time (should converge to 0)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.5)
    
    # Plot 3: Regret distribution (final)
    axes[0, 2].hist(results['regret1'], bins=30, alpha=0.5, label='Player 1')
    axes[0, 2].hist(results['regret2'], bins=30, alpha=0.5, label='Player 2')
    axes[0, 2].set_xlabel('Final Regret')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Final Regret Distribution (MC runs)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Utility distribution
    axes[1, 0].hist(results['utility1'], bins=30, alpha=0.5, label='Player 1')
    axes[1, 0].hist(results['utility2'], bins=30, alpha=0.5, label='Player 2')
    axes[1, 0].set_xlabel('Total Utility')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Total Utility Distribution (MC runs)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Win rate distribution
    axes[1, 1].hist(results['win_rate1'], bins=30, alpha=0.5, label='Player 1')
    axes[1, 1].hist(results['win_rate2'], bins=30, alpha=0.5, label='Player 2')
    axes[1, 1].set_xlabel('Win Rate')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Win Rate Distribution (MC runs)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Bid variance over time (stability measure)
    bid_variance1 = np.var(results['bids1'], axis=0)
    bid_variance2 = np.var(results['bids2'], axis=0)
    axes[1, 2].plot(bid_variance1, label='Player 1', alpha=0.7)
    axes[1, 2].plot(bid_variance2, label='Player 2', alpha=0.7)
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
    print(f"Player 1:")
    print(f"  Mean Regret: {np.mean(results['regret1']):.2f} ± {np.std(results['regret1']):.2f}")
    print(f"  Mean Utility: {np.mean(results['utility1']):.2f} ± {np.std(results['utility1']):.2f}")
    print(f"  Mean Win Rate: {np.mean(results['win_rate1']):.3f} ± {np.std(results['win_rate1']):.3f}")
    print(f"\nPlayer 2:")
    print(f"  Mean Regret: {np.mean(results['regret2']):.2f} ± {np.std(results['regret2']):.2f}")
    print(f"  Mean Utility: {np.mean(results['utility2']):.2f} ± {np.std(results['utility2']):.2f}")
    print(f"  Mean Win Rate: {np.mean(results['win_rate2']):.3f} ± {np.std(results['win_rate2']):.3f}")

