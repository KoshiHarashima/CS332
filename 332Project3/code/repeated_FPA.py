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
    alg1_func, v1_config, env1 = player1_config
    alg2_func, v2_config, env2 = player2_config
    
    # Check if values are callable (functions) or fixed values
    v1_is_callable = callable(v1_config)
    v2_is_callable = callable(v2_config)
    
    # Initialize environment state for each player (will be updated per MC run if callable)
    env1.setdefault('k', k)
    if not v1_is_callable:
        env1.setdefault('h', v1_config)
    env2.setdefault('k', k)
    if not v2_is_callable:
        env2.setdefault('h', v2_config)
    
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
    all_winning_prices = []
    
    # Monte Carlo loop
    for mc_iter in range(n_mc):
        # Reset for each MC run
        history1 = []
        history2 = []
        env1_state = env1.copy()
        env2_state = env2.copy()
        
        # Create independent random number generators for each player
        # This ensures that each player's internal randomness (exploration, tie-breaking in selection)
        # is independent, while maintaining Common Random Numbers (CRN) for exogenous randomness
        # (like tie-breaking in auction outcomes)
        # Use different seeds for each player to ensure independence
        base_seed = mc_iter * 1000  # Different base seed for each MC run
        env1_state['random_state'] = np.random.RandomState(base_seed + 1)
        env2_state['random_state'] = np.random.RandomState(base_seed + 2)
        
        # Common random state for exogenous events (tie-breaking in auction)
        # This ensures fair comparison while maintaining independence in player choices
        common_random_state = np.random.RandomState(base_seed + 999)
        
        env1_state['cumulative_payoffs'] = np.zeros(k)
        env2_state['cumulative_payoffs'] = np.zeros(k)
        
        total_utility1 = 0
        total_utility2 = 0
        wins1 = 0
        wins2 = 0
        
        # Pre-allocate NumPy arrays for better performance (instead of list append)
        bids1_history = np.zeros(n_rounds)
        bids2_history = np.zeros(n_rounds)
        regret_history1 = np.zeros(n_rounds)
        regret_history2 = np.zeros(n_rounds)
        winning_prices = np.zeros(n_rounds)  # Price paid by winner (winner's bid)
        
        # Initialize cumulative utilities for incremental regret calculation
        # Use a unified bid_grid that covers the maximum possible value range
        # For uniform distribution [0, 1.0], we use [0, 1.0] as the grid
        # For fixed values, use max(v1, v2) if known, otherwise use a reasonable default
        if v1_is_callable or v2_is_callable:
            # If using callable values, assume max value is 1.0 (for uniform [0,1])
            # This should match the distribution range used
            max_possible_value = 1.0
        else:
            # Use max of fixed values
            max_possible_value = max(float(v1_config), float(v2_config))
        
        # Unified bid grid for regret calculation (fixed across all rounds)
        unified_bid_grid = np.linspace(0, max_possible_value, k)
        cumulative_fixed_utility1 = np.zeros(k)  # Cumulative utility for each fixed bid for player 1
        cumulative_fixed_utility2 = np.zeros(k)  # Cumulative utility for each fixed bid for player 2
        
        # Store unified_bid_grid in env_state for algorithms to use (ensures consistency across rounds)
        env1_state['unified_bid_grid'] = unified_bid_grid
        env2_state['unified_bid_grid'] = unified_bid_grid
        
        # Pre-compute fixed values if not callable (avoid repeated float conversion)
        if not v1_is_callable:
            v1_fixed = float(v1_config)
        if not v2_is_callable:
            v2_fixed = float(v2_config)
        
        # Round loop
        for round_num in range(n_rounds):
            # Generate values for this round if callable (each round gets new values)
            if v1_is_callable:
                v1 = v1_config()  # Call the function to generate value for this round
                # Ensure v1 is a scalar value, not an array
                if isinstance(v1, np.ndarray):
                    v1 = float(v1.item() if v1.size == 1 else v1[0])
                else:
                    v1 = float(v1)
            else:
                v1 = v1_fixed  # Use pre-computed fixed value
            
            if v2_is_callable:
                v2 = v2_config()  # Call the function to generate value for this round
                # Ensure v2 is a scalar value, not an array
                if isinstance(v2, np.ndarray):
                    v2 = float(v2.item() if v2.size == 1 else v2[0])
                else:
                    v2 = float(v2)
            else:
                v2 = v2_fixed  # Use pre-computed fixed value
            
            # Update h (scaling parameter) based on current round values
            env1_state['h'] = v1
            env2_state['h'] = v2
            
            # Get bids from each player (with current round values)
            bid1 = alg1_func(0, v1, round_num, history1, env1_state)
            bid2 = alg2_func(1, v2, round_num, history2, env2_state)
            
            bids1_history[round_num] = bid1
            bids2_history[round_num] = bid2
            
            # Play FPA
            alloc1, alloc2 = play_fpa_round_2players(bid1, bid2)
            
            # Calculate utilities
            # Utility = allocation * (value - bid)
            utility1 = utility.calculate_utility(v1, alloc1, bid1)
            utility2 = utility.calculate_utility(v2, alloc2, bid2)
            
            # Update totals
            total_utility1 += utility1
            total_utility2 += utility2
            
            # Track wins and determine won status (single random call for tie)
            # Also record winning price (price = winner's bid in first-price auction)
            if alloc1 > 0.5:
                wins1 += 1
                won1 = True
                won2 = False
                winning_prices[round_num] = bid1  # Winner pays their bid
            elif alloc2 > 0.5:
                wins2 += 1
                won1 = False
                won2 = True
                winning_prices[round_num] = bid2  # Winner pays their bid
            else:  # tie
                # Use common random state for fair tie-breaking (exogenous randomness)
                tie_winner = common_random_state.random() < 0.5
                if tie_winner:
                    wins1 += 1
                    won1 = True
                    won2 = False
                    winning_prices[round_num] = bid1  # Winner pays their bid
                else:
                    wins2 += 1
                    won1 = False
                    won2 = True
                    winning_prices[round_num] = bid2  # Winner pays their bid
            # History format: (bid, utility, won, opponent_bid)
            history1.append((bid1, utility1, won1, bid2))
            history2.append((bid2, utility2, won2, bid1))
            
            # Calculate regret INCREMENTALLY: only compute utility for NEW opponent bid
            # Use unified_bid_grid for regret calculation (consistent across rounds)
            # Vectorized computation for better performance
            
            # For player 1: add utility from new opponent bid (bid2) to each fixed bid
            # Vectorized allocation calculation
            valid_mask1 = unified_bid_grid <= v1
            if np.any(valid_mask1):
                # Vectorized comparison for allocation
                rtol = 1e-9
                atol = 1e-9
                fixed_bids_valid = unified_bid_grid[valid_mask1]
                
                # Calculate allocations vectorized
                # Win: fixed_bid > bid2, Lose: fixed_bid < bid2, Tie: fixed_bid == bid2
                greater_mask = fixed_bids_valid > bid2
                less_mask = fixed_bids_valid < bid2
                tie_mask = np.isclose(fixed_bids_valid, bid2, rtol=rtol, atol=atol)
                
                allocations = np.zeros_like(fixed_bids_valid)
                allocations[greater_mask & ~tie_mask] = 1.0
                allocations[tie_mask] = 0.5
                allocations[less_mask & ~tie_mask] = 0.0
                
                # Calculate utilities vectorized: u = allocation * (value - bid)
                new_utilities = allocations * (v1 - fixed_bids_valid)
                cumulative_fixed_utility1[valid_mask1] += new_utilities
            
            # For player 2: add utility from new opponent bid (bid1) to each fixed bid
            # Vectorized allocation calculation
            valid_mask2 = unified_bid_grid <= v2
            if np.any(valid_mask2):
                # Vectorized comparison for allocation
                fixed_bids_valid = unified_bid_grid[valid_mask2]
                
                # Calculate allocations vectorized
                # Win: fixed_bid > bid1, Lose: fixed_bid < bid1, Tie: fixed_bid == bid1
                greater_mask = fixed_bids_valid > bid1
                less_mask = fixed_bids_valid < bid1
                tie_mask = np.isclose(fixed_bids_valid, bid1, rtol=rtol, atol=atol)
                
                allocations = np.zeros_like(fixed_bids_valid)
                allocations[greater_mask & ~tie_mask] = 1.0
                allocations[tie_mask] = 0.5
                allocations[less_mask & ~tie_mask] = 0.0
                
                # Calculate utilities vectorized: u = allocation * (value - bid)
                new_utilities = allocations * (v2 - fixed_bids_valid)
                cumulative_fixed_utility2[valid_mask2] += new_utilities
            
            # Find best fixed bid AFTER updating all fixed bids
            # Use cumulative_fixed_utility directly (already contains utility from all valid rounds)
            # This is much faster than recalculating from scratch (O(k) instead of O(k * n_rounds^2))
            
            # Find best in hindsight utility for player 1
            # cumulative_fixed_utility1 already contains utility from all rounds where fixed_bid <= value
            # For uniform distribution (values change each round), we need to consider only bids that
            # were valid in at least one round. However, since we only add utility when fixed_bid <= value,
            # cumulative_fixed_utility1[i] = 0 means the bid was never valid, so we should ignore it.
            # But actually, if a bid was never valid, it shouldn't be considered as best in hindsight.
            # However, for simplicity, we can just use argmax - if a bid was never valid, its utility is 0,
            # which is less than any valid bid's utility (which should be positive or at least non-negative).
            # But to be safe, we can set invalid bids to -inf before argmax.
            # Actually, wait - if a bid was never valid, cumulative_fixed_utility1[i] = 0, which might
            # be the same as a valid bid that had zero utility. But in practice, valid bids should have
            # positive utility in at least some rounds, so this shouldn't be an issue.
            # However, to handle the case where all bids have zero utility, we should check.
            
            # For uniform distribution: each round has different value, so we need to track which bids
            # were valid in at least one round. But actually, the current implementation is correct:
            # cumulative_fixed_utility1[i] contains the sum of utilities from all rounds where
            # unified_bid_grid[i] <= value in that round. So if a bid was never valid, its cumulative
            # utility is 0. If a bid was valid in some rounds, its cumulative utility is the sum of
            # utilities from those rounds. This is correct for best in hindsight.
            
            # However, there's a subtle issue: if a bid was never valid (cumulative = 0), it might
            # be selected as best if all other bids also have cumulative = 0. But this is fine - if
            # no bid was ever valid, then there's no best in hindsight.
            
            # Actually, the real issue might be that we're comparing cumulative utilities across
            # different sets of rounds. For example, bid=0.1 was valid in more rounds than bid=0.9.
            # But this is correct for best in hindsight - we want the bid that maximizes cumulative
            # utility across all rounds where it was valid.
            
            # So the current implementation should be correct. But let's add a check to ensure we
            # don't select a bid that was never valid (cumulative = 0) when there are valid bids.
            # Actually, this shouldn't happen because valid bids should have positive cumulative utility.
            
            # Let's just use argmax as before, but ensure we handle the case where all utilities are 0
            best_fixed_bid_idx1 = np.argmax(cumulative_fixed_utility1)
            best_in_hindsight_utility1 = cumulative_fixed_utility1[best_fixed_bid_idx1]
            
            # Find best in hindsight utility for player 2
            best_fixed_bid_idx2 = np.argmax(cumulative_fixed_utility2)
            best_in_hindsight_utility2 = cumulative_fixed_utility2[best_fixed_bid_idx2]
            
            # Calculate regret at this round using best in hindsight utility
            regret1_round = best_in_hindsight_utility1 - total_utility1
            regret2_round = best_in_hindsight_utility2 - total_utility2
            regret_history1[round_num] = regret1_round
            regret_history2[round_num] = regret2_round
        
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
        all_winning_prices.append(winning_prices)
        
        if (mc_iter + 1) % 10 == 0:
            print(f"MC iteration {mc_iter + 1}/{n_mc} completed")
    
    # Store v1 and v2 if they are fixed (not callable)
    result_dict = {
        'regret1': np.array(all_regret1),
        'regret2': np.array(all_regret2),
        'utility1': np.array(all_utility1),
        'utility2': np.array(all_utility2),
        'win_rate1': np.array(all_win_rate1),
        'win_rate2': np.array(all_win_rate2),
        'bids1': all_bids1,
        'bids2': all_bids2,
        'regret_history1': all_regret_history1,
        'regret_history2': all_regret_history2,
        'winning_prices': all_winning_prices
    }
    
    # Store values if they are fixed (not callable)
    if not v1_is_callable:
        result_dict['v1'] = float(v1_config)
    if not v2_is_callable:
        result_dict['v2'] = float(v2_config)
    
    return result_dict


def plot_results(results, title="Simulation Results", show_ne_distance=False):
    """
    Plot 5 separate figures and save them as individual PNG files:
    1. Bid evolution over rounds (with optional NE distance)
    2. Regret over time (convergence analysis)
    3. Utility distribution
    4. Win rate distribution
    5. Average 1-round regret convergence (cumulative regret / round number)
    
    Args:
        results: dict with simulation results
        title: title for the plots
        show_ne_distance: if True, show NE distance on bid evolution plot
    """
    figures_dir = Path('../figures')
    figures_dir.mkdir(exist_ok=True)
    
    # Create base filename from title
    base_filename = title.lower().replace(' ', '_').replace(' vs ', '_vs_').replace('(', '').replace(')', '')
    
    # Create a folder for this simulation's figures
    output_folder = figures_dir / base_filename
    output_folder.mkdir(exist_ok=True)
    
    # Get number of MC runs
    n_mc = len(results['regret_history1'])
    
    # Get average bids and regret over MC runs
    bids1_avg = np.mean(results['bids1'], axis=0)
    bids2_avg = np.mean(results['bids2'], axis=0)
    bids1_std = np.std(results['bids1'], axis=0)
    bids2_std = np.std(results['bids2'], axis=0)
    
    # Calculate average regret over time
    regret1_avg = np.mean(results['regret_history1'], axis=0)
    regret2_avg = np.mean(results['regret_history2'], axis=0)
    regret1_std = np.std(results['regret_history1'], axis=0)
    regret2_std = np.std(results['regret_history2'], axis=0)
    
    # Calculate 95% confidence intervals (using standard error)
    # Standard error = std / sqrt(n_mc)
    # 95% CI = mean ± 1.96 * SE (assuming normal distribution)
    bids1_se = bids1_std / np.sqrt(n_mc)
    bids2_se = bids2_std / np.sqrt(n_mc)
    regret1_se = regret1_std / np.sqrt(n_mc)
    regret2_se = regret2_std / np.sqrt(n_mc)
    
    # 95% confidence interval multiplier (1.96 for normal distribution)
    ci_multiplier = 1.96
    
    bids1_ci_lower = bids1_avg - ci_multiplier * bids1_se
    bids1_ci_upper = bids1_avg + ci_multiplier * bids1_se
    bids2_ci_lower = bids2_avg - ci_multiplier * bids2_se
    bids2_ci_upper = bids2_avg + ci_multiplier * bids2_se
    
    regret1_ci_lower = regret1_avg - ci_multiplier * regret1_se
    regret1_ci_upper = regret1_avg + ci_multiplier * regret1_se
    regret2_ci_lower = regret2_avg - ci_multiplier * regret2_se
    regret2_ci_upper = regret2_avg + ci_multiplier * regret2_se
    
    # Plot 1: Bid evolution over rounds
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    rounds_bid = np.arange(len(bids1_avg))
    line1 = ax1.plot(rounds_bid, bids1_avg, label='Player 1', alpha=0.7)
    color1 = line1[0].get_color()
    ax1.fill_between(rounds_bid, bids1_ci_lower, bids1_ci_upper, alpha=0.2, color=color1)
    line2 = ax1.plot(rounds_bid, bids2_avg, label='Player 2', alpha=0.7)
    color2 = line2[0].get_color()
    ax1.fill_between(rounds_bid, bids2_ci_lower, bids2_ci_upper, alpha=0.2, color=color2)
    
    # Add NE distance if requested and values are available
    if show_ne_distance and 'v1' in results and 'v2' in results:
        v1 = results['v1']
        v2 = results['v2']
        ne_bid1 = v1 / 2.0  # Nash equilibrium bid for player 1
        ne_bid2 = v2 / 2.0  # Nash equilibrium bid for player 2
        
        # Calculate NE distance across MC runs for CI
        ne_dist1_all = []
        ne_dist2_all = []
        for mc_run in range(n_mc):
            ne_dist1_run = np.abs(results['bids1'][mc_run] - ne_bid1)
            ne_dist2_run = np.abs(results['bids2'][mc_run] - ne_bid2)
            ne_dist1_all.append(ne_dist1_run)
            ne_dist2_all.append(ne_dist2_run)
        
        ne_dist1_all = np.array(ne_dist1_all)
        ne_dist2_all = np.array(ne_dist2_all)
        
        ne_dist1_mean = np.mean(ne_dist1_all, axis=0)
        ne_dist2_mean = np.mean(ne_dist2_all, axis=0)
        ne_dist1_std = np.std(ne_dist1_all, axis=0)
        ne_dist2_std = np.std(ne_dist2_all, axis=0)
        
        ne_dist1_se = ne_dist1_std / np.sqrt(n_mc)
        ne_dist2_se = ne_dist2_std / np.sqrt(n_mc)
        ne_dist1_ci_lower = ne_dist1_mean - ci_multiplier * ne_dist1_se
        ne_dist1_ci_upper = ne_dist1_mean + ci_multiplier * ne_dist1_se
        ne_dist2_ci_lower = ne_dist2_mean - ci_multiplier * ne_dist2_se
        ne_dist2_ci_upper = ne_dist2_mean + ci_multiplier * ne_dist2_se
        
        # Create second y-axis for NE distance
        ax1_twin = ax1.twinx()
        line3 = ax1_twin.plot(rounds_bid, ne_dist1_mean, label='NE Distance (P1)', 
                              color=color1, linestyle='--', alpha=0.7, linewidth=1.5)
        ax1_twin.fill_between(rounds_bid, ne_dist1_ci_lower, ne_dist1_ci_upper, 
                              alpha=0.15, color=color1)
        line4 = ax1_twin.plot(rounds_bid, ne_dist2_mean, label='NE Distance (P2)', 
                              color=color2, linestyle='--', alpha=0.7, linewidth=1.5)
        ax1_twin.fill_between(rounds_bid, ne_dist2_ci_lower, ne_dist2_ci_upper, 
                              alpha=0.15, color=color2)
        ax1_twin.set_ylabel('NE Distance (|bid - v/2|)', fontsize=10)
        ax1_twin.grid(True, alpha=0.2, linestyle=':')
        
        # Add NE bid lines
        ax1.axhline(y=ne_bid1, color=color1, linestyle=':', alpha=0.5, linewidth=1, 
                   label=f'NE Bid P1 (v/2={ne_bid1:.2f})')
        ax1.axhline(y=ne_bid2, color=color2, linestyle=':', alpha=0.5, linewidth=1, 
                   label=f'NE Bid P2 (v/2={ne_bid2:.2f})')
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_twin.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=9)
    else:
        ax1.legend()
    
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Average Bid')
    ax1.set_title('Bid Evolution Over Rounds' + (' (with NE Distance)' if show_ne_distance and 'v1' in results and 'v2' in results else ''))
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    filepath1 = output_folder / "bid_evolution.png"
    plt.savefig(filepath1, dpi=300, bbox_inches='tight')
    print(f"Plot 1 saved to: {filepath1}")
    plt.close(fig1)
    
    # Plot 2: Regret over time (convergence analysis)
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    rounds = np.arange(len(regret1_avg))
    line1 = ax2.plot(rounds, regret1_avg, label='Player 1', alpha=0.7)
    color1 = line1[0].get_color()
    ax2.fill_between(rounds, regret1_ci_lower, regret1_ci_upper, alpha=0.2, color=color1)
    line2 = ax2.plot(rounds, regret2_avg, label='Player 2', alpha=0.7)
    color2 = line2[0].get_color()
    ax2.fill_between(rounds, regret2_ci_lower, regret2_ci_upper, alpha=0.2, color=color2)
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Average Regret')
    ax2.set_title('Regret Over Time (should converge to 0)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    plt.tight_layout()
    filepath2 = output_folder / "regret.png"
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
    filepath4 = output_folder / "utility_distribution.png"
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
    filepath5 = output_folder / "win_rate_distribution.png"
    plt.savefig(filepath5, dpi=300, bbox_inches='tight')
    print(f"Plot 5 saved to: {filepath5}")
    plt.close(fig5)
    
    # Plot 6: Average 1-round regret convergence
    # Calculate average 1-round regret (cumulative regret / round number)
    regret_history1_array = np.array(results['regret_history1'])  # Shape: (n_mc, n_rounds)
    regret_history2_array = np.array(results['regret_history2'])  # Shape: (n_mc, n_rounds)
    
    n_rounds = regret_history1_array.shape[1]
    rounds_1round = np.arange(1, n_rounds + 1)  # Round numbers (1-indexed)
    
    # Calculate average 1-round regret for each MC run
    # Average 1-round regret at round t = cumulative regret[t] / (t+1)
    avg_1round_regret1 = regret_history1_array / rounds_1round[np.newaxis, :]  # Shape: (n_mc, n_rounds)
    avg_1round_regret2 = regret_history2_array / rounds_1round[np.newaxis, :]  # Shape: (n_mc, n_rounds)
    
    # Calculate mean and std across MC runs
    avg_1round_regret1_mean = np.mean(avg_1round_regret1, axis=0)
    avg_1round_regret2_mean = np.mean(avg_1round_regret2, axis=0)
    avg_1round_regret1_std = np.std(avg_1round_regret1, axis=0)
    avg_1round_regret2_std = np.std(avg_1round_regret2, axis=0)
    
    # Calculate 95% confidence intervals
    avg_1round_regret1_se = avg_1round_regret1_std / np.sqrt(n_mc)
    avg_1round_regret2_se = avg_1round_regret2_std / np.sqrt(n_mc)
    avg_1round_regret1_ci_lower = avg_1round_regret1_mean - ci_multiplier * avg_1round_regret1_se
    avg_1round_regret1_ci_upper = avg_1round_regret1_mean + ci_multiplier * avg_1round_regret1_se
    avg_1round_regret2_ci_lower = avg_1round_regret2_mean - ci_multiplier * avg_1round_regret2_se
    avg_1round_regret2_ci_upper = avg_1round_regret2_mean + ci_multiplier * avg_1round_regret2_se
    
    # Plot average 1-round regret convergence
    fig6, ax6 = plt.subplots(figsize=(10, 6))
    
    # Plot Player 1
    line1 = ax6.plot(rounds_1round, avg_1round_regret1_mean, label='Player 1', alpha=0.8, linewidth=2)
    color1 = line1[0].get_color()
    ax6.fill_between(rounds_1round, avg_1round_regret1_ci_lower, avg_1round_regret1_ci_upper, 
                    alpha=0.2, color=color1, label='Player 1 95% CI')
    
    # Plot Player 2
    line2 = ax6.plot(rounds_1round, avg_1round_regret2_mean, label='Player 2', alpha=0.8, linewidth=2)
    color2 = line2[0].get_color()
    ax6.fill_between(rounds_1round, avg_1round_regret2_ci_lower, avg_1round_regret2_ci_upper, 
                    alpha=0.2, color=color2, label='Player 2 95% CI')
    
    ax6.set_xlabel('Round', fontsize=12)
    ax6.set_ylabel('Average 1-Round Regret', fontsize=12)
    ax6.set_title('Average 1-Round Regret Convergence', fontsize=14)
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)
    ax6.axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=1)
    
    # Use log scale for better visualization of convergence
    ax6.set_yscale('log')
    ax6.set_xscale('log')
    
    plt.tight_layout()
    filepath6 = output_folder / "avg_1round_regret_convergence.png"
    plt.savefig(filepath6, dpi=300, bbox_inches='tight')
    print(f"Plot 6 saved to: {filepath6}")
    plt.close(fig6)
    
    print(f"\nAll figures saved to folder: {output_folder}")


def plot_quantitative_metrics(results, title="Quantitative Metrics", save_dir=None):
    """
    Plot quantitative metrics: utility, price, and regret for exploitation analysis.
    
    Args:
        results: dict with simulation results
        title: title for the plots
        save_dir: directory to save plots (default: ../figures)
    """
    if save_dir is None:
        save_dir = Path('../figures')
    save_dir.mkdir(exist_ok=True)
    
    base_filename = title.lower().replace(' ', '_').replace(' vs ', '_vs_').replace('(', '').replace(')', '')
    n_mc = len(results['regret_history1'])
    
    # Calculate metrics
    utility1_mean = np.mean(results['utility1'])
    utility2_mean = np.mean(results['utility2'])
    utility1_std = np.std(results['utility1'])
    utility2_std = np.std(results['utility2'])
    
    regret1_mean = np.mean(results['regret1'])
    regret2_mean = np.mean(results['regret2'])
    regret1_std = np.std(results['regret1'])
    regret2_std = np.std(results['regret2'])
    
    # Calculate average winning price
    all_prices = []
    for prices in results['winning_prices']:
        all_prices.extend(prices)
    price_mean = np.mean(all_prices)
    price_std = np.std(all_prices)
    
    # Calculate 95% CI
    ci_multiplier = 1.96
    utility1_se = utility1_std / np.sqrt(n_mc)
    utility2_se = utility2_std / np.sqrt(n_mc)
    regret1_se = regret1_std / np.sqrt(n_mc)
    regret2_se = regret2_std / np.sqrt(n_mc)
    price_se = price_std / np.sqrt(len(results['winning_prices']) * len(results['winning_prices'][0]))
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Utility
    ax1 = axes[0]
    x_pos = np.arange(2)
    utilities = [utility1_mean, utility2_mean]
    errors = [ci_multiplier * utility1_se, ci_multiplier * utility2_se]
    bars1 = ax1.bar(x_pos, utilities, yerr=errors, capsize=5, alpha=0.7, 
                    color=['#1f77b4', '#ff7f0e'], edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Total Utility', fontsize=12)
    ax1.set_xlabel('Player', fontsize=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(['Player 1\n(Flexible)', 'Player 2\n(Exploitation)'])
    ax1.set_title('Total Utility', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    # Add value labels on bars
    for i, (bar, val, err) in enumerate(zip(bars1, utilities, errors)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + err + height*0.02,
                f'{val:.1f}±{err:.1f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 2: Average Winning Price
    ax2 = axes[1]
    bars2 = ax2.bar([0], [price_mean], yerr=[ci_multiplier * price_se], capsize=5, 
                    alpha=0.7, color='#2ca02c', edgecolor='black', linewidth=1.5, width=0.6)
    ax2.set_ylabel('Average Winning Price', fontsize=12)
    ax2.set_xlabel('')
    ax2.set_xticks([0])
    ax2.set_xticklabels([''])
    ax2.set_title('Average Winning Price', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    # Add value label
    ax2.text(0, price_mean + ci_multiplier * price_se + price_mean*0.02,
            f'{price_mean:.3f}±{ci_multiplier * price_se:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Plot 3: Regret
    ax3 = axes[2]
    regrets = [regret1_mean, regret2_mean]
    errors_regret = [ci_multiplier * regret1_se, ci_multiplier * regret2_se]
    bars3 = ax3.bar(x_pos, regrets, yerr=errors_regret, capsize=5, alpha=0.7,
                    color=['#1f77b4', '#ff7f0e'], edgecolor='black', linewidth=1.5)
    ax3.set_ylabel('Total Regret', fontsize=12)
    ax3.set_xlabel('Player', fontsize=12)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(['Player 1\n(Flexible)', 'Player 2\n(Exploitation)'])
    ax3.set_title('Total Regret', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    # Add value labels on bars
    for i, (bar, val, err) in enumerate(zip(bars3, regrets, errors_regret)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + err + height*0.02,
                f'{val:.1f}±{err:.1f}', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    filepath = save_dir / f"{base_filename}_quantitative_metrics.png"
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Quantitative metrics plot saved to: {filepath}")
    plt.close(fig)
    
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
    data_dir = Path(output_dir)
    data_dir.mkdir(exist_ok=True)
    
    # Create base filename from title
    base_filename = title.lower().replace(' ', '_').replace(' vs ', '_vs_').replace('(', '').replace(')', '')
    
    # Create a folder for this simulation's data
    output_path = data_dir / base_filename
    output_path.mkdir(exist_ok=True)
    
    # Use base_filename for individual file names
    filename = base_filename
    
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
    
    print(f"\nAll data files saved to folder: {output_path}")

