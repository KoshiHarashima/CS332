# visualization of results
# we visualize the results of the Monte Carlo simulation

# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Set up paths dynamically
current_dir = Path(__file__).parent
project_root = current_dir.parent
data_dir = project_root / 'data'
figures_dir = project_root / 'figures'

# Create figures directory if it doesn't exist
figures_dir.mkdir(exist_ok=True)

# Set up the plotting style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Define colors for different epsilon types
colors = {'random': 'blue', 'optimal': 'red', 'FTL': 'green'}
line_styles = {'random': '-', 'optimal': '--', 'FTL': '-.'}

def plot_regret_comparison(df, title, filename):
    """
    Plot regret comparison with mean and confidence intervals for different epsilon values
    """
    plt.figure(figsize=(12, 8))
    
    for epsilon_type in ['random', 'optimal', 'FTL']:
        subset = df[df['epsilon_type'] == epsilon_type]
        
        # Calculate confidence interval (mean ± std)
        mean_regret = subset['mean_regret'].values
        std_regret = subset['std_regret'].values
        rounds = subset['round'].values
        
        # Plot mean regret
        plt.plot(rounds, mean_regret, 
                color=colors[epsilon_type], 
                linestyle=line_styles[epsilon_type],
                linewidth=2, 
                label=f'{epsilon_type} (ε={subset["epsilon_value"].iloc[0]:.4f})')
        
        # Plot confidence interval
        plt.fill_between(rounds, 
                        mean_regret - std_regret, 
                        mean_regret + std_regret,
                        color=colors[epsilon_type], 
                        alpha=0.2)
    
    plt.xlabel('Round Number')
    plt.ylabel('Regret')
    plt.title(f'Regret Comparison: {title}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    save_path = figures_dir / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
def plot_payoff_comparison(df, title, filename, max_rounds=None):
    """
    Plot payoff comparison with mean and confidence intervals for different epsilon values
    (Similar to plot_regret_comparison but for payoff data)
    
    Args:
        df: DataFrame with payoff data
        title: Plot title
        filename: Output filename
        max_rounds: Maximum number of rounds to display (None for all rounds)
    """
    plt.figure(figsize=(12, 8))
    
    # Filter data to max_rounds if specified
    if max_rounds is not None:
        df_filtered = df[df['round'] <= max_rounds]
        print(f"Filtered data to first {max_rounds} rounds: {df_filtered.shape[0]} rows")
    else:
        df_filtered = df
    
    # Debug: Print data info
    print(f"Data shape: {df_filtered.shape}")
    print(f"Columns: {list(df_filtered.columns)}")
    print(f"Epsilon types: {df_filtered['epsilon_type'].unique()}")
    
    for epsilon_type in ['random', 'optimal', 'FTL']:
        subset = df_filtered[df_filtered['epsilon_type'] == epsilon_type]
        
        if len(subset) == 0:
            print(f"Warning: No data found for epsilon_type '{epsilon_type}'")
            continue
            
        # Calculate confidence interval (mean ± std)
        mean_payoff = subset['mean_payoff'].values
        std_payoff = subset['std_payoff'].values
        rounds = subset['round'].values
        
        print(f"Epsilon type: {epsilon_type}, Data points: {len(rounds)}")
        print(f"Mean payoff range: {mean_payoff.min():.4f} to {mean_payoff.max():.4f}")
        
        # Plot mean payoff
        plt.plot(rounds, mean_payoff, 
                color=colors[epsilon_type], 
                linestyle=line_styles[epsilon_type],
                linewidth=2, 
                label=f'{epsilon_type} (ε={subset["epsilon_value"].iloc[0]:.4f})')
        
        # Plot confidence interval
        plt.fill_between(rounds, 
                        mean_payoff - std_payoff, 
                        mean_payoff + std_payoff,
                        color=colors[epsilon_type], 
                        alpha=0.2)
    
    plt.xlabel('Round Number', fontsize=12)
    plt.ylabel('Cumulative Payoff', fontsize=12)
    plot_title = f'Cumulative Payoff Comparison: {title}'
    if max_rounds is not None:
        plot_title += f' (First {max_rounds} Rounds)'
    plt.title(plot_title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    save_path = figures_dir / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved to: {save_path}")
