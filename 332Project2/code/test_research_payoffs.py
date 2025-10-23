#!/usr/bin/env python3
"""
Test script for Research Payoffs with Fixed-Share Exponentiated Gradient Algorithm.

This script demonstrates the new implementation with:
- Fixed cluster structure (MIT, Northwestern, Stanford)
- Regime switching with different leading actions
- Fixed-Share Exponentiated Gradient learning
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add code directory to path
sys.path.append(str(Path(__file__).parent))

from D_rp import ResearchPayoffs
from FixedShareEG import FixedShareEG

def test_research_payoffs():
    """Test the Research Payoffs environment."""
    print("Testing Research Payoffs Environment")
    print("=" * 50)
    
    # Initialize environment
    env = ResearchPayoffs(k=10, mu=0.5, delta=0.3, rho=0.5, h=0.1, sigma=0.1, seed=42)
    
    print(f"Clusters: {env.clusters}")
    print(f"Leaders: {env.leaders}")
    print(f"Initial regime: {env.current_regime}")
    print()
    
    # Test regime switching
    print("Testing regime switching:")
    for i in range(10):
        alpha = env.generate_payoffs(i)
        regime = env.get_current_regime()
        leader = env.get_leader_action()
        print(f"Round {i+1}: Regime {regime}, Leader action {leader}, Alpha[leader] = {alpha[leader]:.4f}")
    
    print()

def test_fixed_share_eg():
    """Test the Fixed-Share Exponentiated Gradient algorithm."""
    print("Testing Fixed-Share Exponentiated Gradient Algorithm")
    print("=" * 60)
    
    # Parameters
    k = 10
    n = 1000
    epsilon = np.sqrt(2 * np.log(k) / n)  # ε_t = √(2 log k / t)
    lambda_param = 0.1  # λ ≈ 1/L where L = 1/h = 10
    
    # Initialize environment and algorithm
    env = ResearchPayoffs(k=k, mu=0.5, delta=0.3, rho=0.5, h=0.1, sigma=0.1, seed=42)
    algorithm = FixedShareEG(k=k, epsilon=epsilon, n=n, lambda_param=lambda_param)
    
    print(f"Parameters:")
    print(f"  k = {k}")
    print(f"  n = {n}")
    print(f"  ε = {epsilon:.6f}")
    print(f"  λ = {lambda_param}")
    print()
    
    # Run algorithm
    regret_history, total_payoff, cumulative_payoffs, payoff_history = algorithm.run_algorithm(env.generate_payoffs)
    
    print(f"Results:")
    print(f"  Total payoff: {total_payoff:.4f}")
    print(f"  Final regret: {regret_history[-1]:.4f}")
    print(f"  Max regret: {np.max(regret_history):.4f}")
    print()
    
    # Analyze weight evolution
    weight_history = algorithm.get_weight_history()
    print("Weight evolution analysis:")
    for action in range(k):
        final_weight = weight_history[-1, action]
        max_weight = np.max(weight_history[:, action])
        print(f"  Action {action}: Final weight = {final_weight:.4f}, Max weight = {max_weight:.4f}")
    
    print()
    
    return regret_history, payoff_history, weight_history

def plot_results(regret_history, payoff_history, weight_history):
    """Plot the results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Regret progression
    axes[0, 0].plot(regret_history)
    axes[0, 0].set_title('Cumulative Regret Progression')
    axes[0, 0].set_xlabel('Round')
    axes[0, 0].set_ylabel('Cumulative Regret')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Payoff progression
    axes[0, 1].plot(payoff_history)
    axes[0, 1].set_title('Cumulative Payoff Progression')
    axes[0, 1].set_xlabel('Round')
    axes[0, 1].set_ylabel('Cumulative Payoff')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Weight evolution for each action
    for action in range(10):
        axes[1, 0].plot(weight_history[:, action], label=f'Action {action}', alpha=0.7)
    axes[1, 0].set_title('Weight Evolution for All Actions')
    axes[1, 0].set_xlabel('Round')
    axes[1, 0].set_ylabel('Weight')
    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Final weight distribution
    final_weights = weight_history[-1, :]
    axes[1, 1].bar(range(10), final_weights)
    axes[1, 1].set_title('Final Weight Distribution')
    axes[1, 1].set_xlabel('Action')
    axes[1, 1].set_ylabel('Weight')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    figures_dir = Path(__file__).parent.parent / 'figures'
    figures_dir.mkdir(exist_ok=True)
    plt.savefig(figures_dir / 'research_payoffs_fixed_share_test.png', dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {figures_dir / 'research_payoffs_fixed_share_test.png'}")
    
    plt.show()

def main():
    """Main test function."""
    print("Research Payoffs with Fixed-Share Exponentiated Gradient Algorithm")
    print("=" * 70)
    print()
    
    # Test environment
    test_research_payoffs()
    
    # Test algorithm
    regret_history, payoff_history, weight_history = test_fixed_share_eg()
    
    # Plot results
    plot_results(regret_history, payoff_history, weight_history)
    
    print("Test completed successfully!")

if __name__ == "__main__":
    main()
