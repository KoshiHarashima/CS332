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

# Main execution (only when run as script, not when imported)
if __name__ == '__main__':
    # Load the results from CSV files
    adversarial_df = pd.read_csv(data_dir / 'adversarial_fair_payoffs_results.csv')
    bernoulli_df = pd.read_csv(data_dir / 'bernoulli_payoffs_results.csv')
    
    # Plot AdversarialFairPayoffs results
    plot_regret_comparison(adversarial_df, 'AdversarialFairPayoffs Environment', 'adversarial_regret_comparison.png')
    
    # Plot BernoulliPayoffs results  
    plot_regret_comparison(bernoulli_df, 'BernoulliPayoffs Environment', 'bernoulli_regret_comparison.png')
    
    print("Visualization completed! Plots saved to figures/ directory.")
