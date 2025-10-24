#!/usr/bin/env python
# coding: utf-8

# In[4]:


# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path


# In[5]:


# Set up paths dynamically
current_dir = Path.cwd()
project_root = current_dir.parent if current_dir.name == 'code' else current_dir
code_dir = project_root / 'code'
data_dir = project_root / 'data'
figures_dir = project_root / 'figures'

# Create directories if they don't exist
data_dir.mkdir(exist_ok=True)
figures_dir.mkdir(exist_ok=True)

# Add code directory to Python path
sys.path.append(str(code_dir))

print(f"Project root: {project_root}")
print(f"Code directory: {code_dir}")
print(f"Data directory: {data_dir}")
print(f"Figures directory: {figures_dir}")


# In[6]:


# Import custom modules
from A_afp import AdversarialFairPayoffs
from B_bp import BernoulliPayoffs
from C_pp import EspacePayoffs
from D_rp import ResearchPayoffs
from EW import ExponentialWeights
from MC import MonteCarloSimulation


# In[7]:


# Fixed parameters
k = 10
n = 1000
num_simulations = 100  # Monte Carlo simulation times

# Epsilon values
epsilon_values = {
    'random': 0.01,  
    'optimal': np.sqrt(np.log(k) / n), 
    'FTL': 100  
}


# # C. 
# 

# In[8]:


# Load normalized data
normalized_data = pd.read_csv(data_dir / 'espace_5stores_normalized.csv')
print("Normalized data shape:", normalized_data.shape)
print("\nNormalized data preview:")
print(normalized_data.head())
print("\nStore statistics after normalization:")
print(normalized_data.groupby('store')['roi_final'].agg(['mean', 'std', 'min', 'max']).round(6))


# In[9]:


# Modified EspacePayoffs class for normalized data
class NormalizedEspacePayoffs:
    """PayoffGenerator using normalized ROI data from 5 Espace stores"""

    def __init__(self, k):
        """
        Initialize normalized ROI-based payoff generator

        Args:
            k: Number of stores (5 for Espace stores)
        """
        self.data = normalized_data.copy()
        self.stores = self.data['store'].unique()
        self.k = len(self.stores)

        # State variables
        self.current_day = 0
        self.total_days = len(self.data)

        print(f"Initialized with {self.k} stores: {list(self.stores)}")
        print(f"Total days available: {self.total_days}")

    def generate_payoffs(self, round_num):
        """Generate payoffs for each store using normalized ROI data"""
        if self.current_day >= self.total_days:
            # If data is exhausted, repeat the last day
            day_data = self.data.iloc[-1]
        else:
            day_data = self.data.iloc[self.current_day]

        # Create payoffs array for all stores
        payoffs = np.zeros(self.k)

        # Get ROI for each store on this day
        for i, store in enumerate(self.stores):
            store_data = self.data[(self.data['store'] == store) & (self.data['day'] == day_data['day'])]
            if len(store_data) > 0:
                payoffs[i] = store_data['roi_final'].iloc[0]
            else:
                # If no data for this store on this day, use the store's mean
                store_mean = self.data[self.data['store'] == store]['roi_final'].mean()
                payoffs[i] = store_mean

        self.current_day += 1
        return payoffs

    def reset(self):
        """Reset state"""
        self.current_day = 0


# In[ ]:


# Run EW algorithm on normalized Espace data
k_espace = 5  # 5 stores
n_espace = 365  # 1 year of data
num_simulations_espace = 1000 # Reduced for faster execution

# Epsilon values for Espace data
epsilon_values_espace = {
    'random': 0.01,
    'optimal': np.sqrt(np.log(k_espace) / n_espace),
    'FTL': 100
}

print(f"Espace EW Algorithm Parameters:")
print(f"  k (stores): {k_espace}")
print(f"  n (rounds): {n_espace}")
print(f"  Simulations: {num_simulations_espace}")
print(f"  Epsilon values: {epsilon_values_espace}")
print()

# Run simulations
espace_results = {}
espace_payoff_progression = {}

for epsilon_name, epsilon_value in epsilon_values_espace.items():
    print(f"Running Espace simulations for {epsilon_name} (epsilon = {epsilon_value:.6f})...")

    regret_histories = []
    total_payoffs = []
    payoff_histories = []

    for sim in range(num_simulations_espace):
        env = NormalizedEspacePayoffs(k_espace)
        algorithm = ExponentialWeights(k_espace, epsilon=epsilon_value, n=n_espace)

        # Run algorithm
        regret_history, total_payoff, cumulative_payoffs, payoff_history = algorithm.run_algorithm(env.generate_payoffs)

        regret_histories.append(regret_history)
        total_payoffs.append(total_payoff)
        payoff_histories.append(payoff_history)

    # Calculate statistics
    mean_regret = np.mean(regret_histories, axis=0)
    std_regret = np.std(regret_histories, axis=0)
    mean_total_payoff = np.mean(total_payoffs)
    std_total_payoff = np.std(total_payoffs)

    espace_results[epsilon_name] = {
        'mean_regret': mean_regret,
        'std_regret': std_regret,
        'mean_total_payoff': mean_total_payoff,
        'std_total_payoff': std_total_payoff,
        'regret_histories': regret_histories,
        'total_payoffs': total_payoffs
    }

    espace_payoff_progression[epsilon_name] = payoff_histories

    print(f"  Mean final regret: {mean_regret[-1]:.4f} ± {std_regret[-1]:.4f}")
    print(f"  Mean total payoff: {mean_total_payoff:.4f} ± {std_total_payoff:.4f}")

print("\nEspace EW Algorithm simulation completed!")


# In[11]:


# Visualize Espace EW Algorithm results
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Espace 5 Stores - EW Algorithm Performance', fontsize=16, fontweight='bold')

# 1. Regret comparison
axes[0,0].set_title('Regret Comparison')
for epsilon_name in epsilon_values_espace.keys():
    mean_regret = espace_results[epsilon_name]['mean_regret']
    std_regret = espace_results[epsilon_name]['std_regret']
    rounds = np.arange(1, len(mean_regret) + 1)

    axes[0,0].plot(rounds, mean_regret, label=f'{epsilon_name}', linewidth=2)
    axes[0,0].fill_between(rounds, mean_regret - std_regret, mean_regret + std_regret, alpha=0.3)

axes[0,0].set_xlabel('Round')
axes[0,0].set_ylabel('Regret')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# 2. Payoff progression
axes[0,1].set_title('Payoff Progression')
for epsilon_name in epsilon_values_espace.keys():
    payoff_histories = espace_payoff_progression[epsilon_name]
    mean_payoff = np.mean(payoff_histories, axis=0)
    std_payoff = np.std(payoff_histories, axis=0)
    rounds = np.arange(1, len(mean_payoff) + 1)

    axes[0,1].plot(rounds, mean_payoff, label=f'{epsilon_name}', linewidth=2)
    axes[0,1].fill_between(rounds, mean_payoff - std_payoff, mean_payoff + std_payoff, alpha=0.3)

axes[0,1].set_xlabel('Round')
axes[0,1].set_ylabel('Cumulative Payoff')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# 3. Final regret distribution
axes[1,0].set_title('Final Regret Distribution')
final_regrets = []
labels = []
for epsilon_name in epsilon_values_espace.keys():
    regret_histories = espace_results[epsilon_name]['regret_histories']
    final_regret = [history[-1] for history in regret_histories]
    final_regrets.append(final_regret)
    labels.append(epsilon_name)

axes[1,0].boxplot(final_regrets, labels=labels)
axes[1,0].set_ylabel('Final Regret')
axes[1,0].grid(True, alpha=0.3)

# 4. Total payoff distribution
axes[1,1].set_title('Total Payoff Distribution')
total_payoffs = []
for epsilon_name in epsilon_values_espace.keys():
    total_payoffs.append(espace_results[epsilon_name]['total_payoffs'])

axes[1,1].boxplot(total_payoffs, labels=labels)
axes[1,1].set_ylabel('Total Payoff')
axes[1,1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(figures_dir / 'espace_ew_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("Espace EW Algorithm visualization completed!")


# In[ ]:


# Detailed analysis of Espace store performance
print("=== ESPACE STORE PERFORMANCE ANALYSIS ===")
print()

# Analyze which stores are most frequently selected
store_selection_analysis = {}

for epsilon_name in epsilon_values_espace.keys():
    print(f"--- {epsilon_name.upper()} EPSILON ---")

    # Get algorithm runs
    regret_histories = espace_results[epsilon_name]['regret_histories']

    # Analyze store selection patterns (simplified - would need access to action histories)
    mean_final_regret = np.mean([history[-1] for history in regret_histories])
    std_final_regret = np.std([history[-1] for history in regret_histories])

    print(f"Final Regret: {mean_final_regret:.4f} ± {std_final_regret:.4f}")

    # Calculate theoretical bounds
    if epsilon_name == 'optimal':
        theoretical_bound = np.sqrt(2 * np.log(k_espace) * n_espace)
        print(f"Theoretical bound: {theoretical_bound:.4f}")
        print(f"Actual/Theory ratio: {mean_final_regret / theoretical_bound:.4f}")

    print()

# Store-specific analysis using normalized data
print("=== STORE-SPECIFIC CHARACTERISTICS ===")
store_stats = normalized_data.groupby('store')['roi_final'].agg(['mean', 'std', 'min', 'max']).round(6)
print(store_stats)

# Calculate store performance ranking
store_means = normalized_data.groupby('store')['roi_final'].mean()
store_stds = normalized_data.groupby('store')['roi_final'].std()
store_sharpe = store_means / store_stds

print(f"\nStore Performance Ranking (by Sharpe ratio):")
store_ranking = store_sharpe.sort_values(ascending=False)
for i, (store, sharpe) in enumerate(store_ranking.items(), 1):
    print(f"{i}. {store}: {sharpe:.4f}")

print(f"\nEspace EW Algorithm analysis completed!")


# # A

# In[11]:


# AdversarialFairPayoffs environment

# Save results
results_data = []
payoff_progression_data = {}  # Store payoff progression data for plotting

for epsilon_name, epsilon_value in epsilon_values.items():
    print(f"\nRunning simulations for {epsilon_name} (epsilon = {epsilon_value:.6f})...")

    regret_histories = []
    total_payoffs = []
    payoff_histories = []  # Store payoff histories for progression analysis

    for sim in range(num_simulations):
        env = AdversarialFairPayoffs(k)
        algorithm = ExponentialWeights(k, epsilon=epsilon_value, n=n)

        # Run algorithm
        regret_history, total_payoff, cumulative_payoffs, payoff_history = algorithm.run_algorithm(env.generate_payoffs)

        regret_histories.append(regret_history)
        total_payoffs.append(total_payoff)
        payoff_histories.append(payoff_history)  # Store payoff progression

        if (sim + 1) % 20 == 0:
            print(f"  Completed {sim + 1}/{num_simulations} simulations")

    # Store payoff progression data for this epsilon type
    payoff_progression_data[epsilon_name] = {
        'rounds': list(range(1, n + 1)),
        'mean_payoff': np.mean(payoff_histories, axis=0).tolist(),
        'std_payoff': np.std(payoff_histories, axis=0).tolist(),
        'epsilon_value': epsilon_value
    }

    # Add results to dataframe
    regret_array = np.array(regret_histories)
    payoff_array = np.array(payoff_histories)
    mean_regret = np.mean(regret_array, axis=0)
    std_regret = np.std(regret_array, axis=0)
    mean_payoff = np.mean(payoff_array, axis=0)
    std_payoff = np.std(payoff_array, axis=0)
    final_regrets = regret_array[:, -1]
    final_payoffs = payoff_array[:, -1]

    for round_num in range(n):
        results_data.append({
            'round': round_num + 1,
            'epsilon_type': epsilon_name,
            'epsilon_value': epsilon_value,
            'mean_regret': mean_regret[round_num],
            'std_regret': std_regret[round_num],
            'mean_payoff': mean_payoff[round_num],
            'std_payoff': std_payoff[round_num],
            'final_regret_mean': np.mean(final_regrets),
            'final_regret_std': np.std(final_regrets),
            'final_payoff_mean': np.mean(final_payoffs),
            'final_payoff_std': np.std(final_payoffs)
        })

print("\nAll simulations completed!")

results_df = pd.DataFrame(results_data)

# Save results to csv file
adversarial_csv_path = data_dir / 'adversarial_fair_payoffs_results.csv'
results_df.to_csv(adversarial_csv_path, index=False)
print(f"Results saved to {adversarial_csv_path}")

# Statistical results
print("\nFinal regret statistics:")
for epsilon_name in research_eg_df["epsilon_type"].unique():
    subset = results_df[results_df['epsilon_type'] == epsilon_name]
    final_stats = subset.iloc[0]  
    print(f"  {epsilon_name}: Mean = {final_stats['final_regret_mean']:.4f}, Std = {final_stats['final_regret_std']:.4f}")


# In[6]:


# Import visualization functions and run plots
import visualization

# Load the data
adversarial_df = pd.read_csv(adversarial_csv_path)

# Use the plot_regret_comparison function from visualization.py
visualization.plot_regret_comparison(adversarial_df, 'Adversarial Fair Payoffs', 'adversarial_regret_comparison.png')

# Plot payoff progression if data is available
visualization.plot_payoff_comparison(adversarial_df, "Adversarial Fair Payoffs", "afp_payoff_progression.png")


# # B

# In[13]:


# BernoulliPayoffs environment

# Save results
results_data_bernoulli = []
payoff_progression_data_bernoulli = {}  # Store payoff progression data for plotting

for epsilon_name, epsilon_value in epsilon_values.items():
    print(f"\nRunning simulations for {epsilon_name} (epsilon = {epsilon_value:.6f})...")

    regret_histories = []
    total_payoffs = []
    payoff_histories = []  # Store payoff histories for progression analysis

    for sim in range(num_simulations):
        env = BernoulliPayoffs(k)
        algorithm = ExponentialWeights(k, epsilon=epsilon_value, n=n)

        # Run algorithm
        regret_history, total_payoff, cumulative_payoffs, payoff_history = algorithm.run_algorithm(env.generate_payoffs)

        regret_histories.append(regret_history)
        total_payoffs.append(total_payoff)
        payoff_histories.append(payoff_history)  # Store payoff progression

        if (sim + 1) % 20 == 0:
            print(f"  Completed {sim + 1}/{num_simulations} simulations")

    # Store payoff progression data for this epsilon type
    payoff_progression_data_bernoulli[epsilon_name] = {
        'rounds': list(range(1, n + 1)),
        'mean_payoff': np.mean(payoff_histories, axis=0).tolist(),
        'std_payoff': np.std(payoff_histories, axis=0).tolist(),
        'epsilon_value': epsilon_value
    }

    # Add results to dataframe
    regret_array = np.array(regret_histories)
    payoff_array = np.array(payoff_histories)
    mean_regret = np.mean(regret_array, axis=0)
    std_regret = np.std(regret_array, axis=0)
    mean_payoff = np.mean(payoff_array, axis=0)
    std_payoff = np.std(payoff_array, axis=0)
    final_regrets = regret_array[:, -1]
    final_payoffs = payoff_array[:, -1]

    for round_num in range(n):
        results_data_bernoulli.append({
            'round': round_num + 1,
            'epsilon_type': epsilon_name,
            'epsilon_value': epsilon_value,
            'mean_regret': mean_regret[round_num],
            'std_regret': std_regret[round_num],
            'mean_payoff': mean_payoff[round_num],
            'std_payoff': std_payoff[round_num],
            'final_regret_mean': np.mean(final_regrets),
            'final_regret_std': np.std(final_regrets),
            'final_payoff_mean': np.mean(final_payoffs),
            'final_payoff_std': np.std(final_payoffs)
        })

print("\nAll simulations completed!")

results_df_bernoulli = pd.DataFrame(results_data_bernoulli)

# Save results to csv file
bernoulli_csv_path = data_dir / 'bernoulli_payoffs_results.csv'
results_df_bernoulli.to_csv(bernoulli_csv_path, index=False)
print(f"Results saved to {bernoulli_csv_path}")

# Statistical results
print("\nFinal regret statistics:")
for epsilon_name in research_eg_df["epsilon_type"].unique():
    subset = results_df_bernoulli[results_df_bernoulli['epsilon_type'] == epsilon_name]
    final_stats = subset.iloc[0]  
    print(f"  {epsilon_name}: Mean = {final_stats['final_regret_mean']:.4f}, Std = {final_stats['final_regret_std']:.4f}")


# In[ ]:


# Import visualization functions and run plots
import visualization

# Load the data
bernoulli_df = pd.read_csv(bernoulli_csv_path)

# Use the plot_regret_comparison function from visualization.py
visualization.plot_regret_comparison(bernoulli_df, 'BernoulliPayoffs Environment', 'bernoulli_regret_comparison.png')

# Plot payoff progression from CSV data
visualization.plot_payoff_comparison(bernoulli_df, "Bernoulli Payoffs", "bernoulli_payoff_progression.png")


# # C

# In[ ]:





# # D

# In[14]:


# D - Research Payoffs with Cluster-based Paper Selection and Markov Regime Switching

# Import Exponentiated Gradient
from EG import ExponentiatedGradientFI

# Parameters for Research Payoffs (new specification)
num_clusters = 4  # Number of clusters (3 or more)
papers_per_cluster = 8  # Papers per cluster
total_papers = num_clusters * papers_per_cluster  # Total number of papers
selected_papers = num_clusters * 3  # Fixed: 3 papers per cluster
n_research = 1000

# Epsilon values for Research Payoffs (using sqrt(2*log(k)/t) formula)
epsilon_values_research = {
    'random': 0.01,
    'optimal': np.sqrt(2 * np.log(selected_papers) / n_research),  # ε_t = √(2 log k / t) where k = selected_papers
    'FTL': 100.0  # FTL uses large epsilon
}

# Initialize Research Payoffs model (new specification)
research_model = ResearchPayoffs(
    num_clusters=num_clusters,
    papers_per_cluster=papers_per_cluster,
    regime_switch_prob=0.3,  # 30% chance to switch regime
    seed=42
)

print("Research Payoffs Environment Info:")
print(f"  Number of clusters: {research_model.num_clusters}")
print(f"  Papers per cluster: {research_model.papers_per_cluster}")
print(f"  Total papers: {research_model.total_papers}")
print(f"  Selected papers per round: {research_model.papers_per_round}")
print(f"  Current regime: {research_model.current_regime}")
# Get cluster information for researcher
cluster_info = research_model.get_cluster_info_for_researcher()
print(f"  Cluster mapping: {cluster_info['cluster_mapping']}")
print()

# Run simulations for each epsilon value
research_results = []
results_data_research = []
payoff_progression_data_research = {}  # Store payoff progression data for plotting

# Initialize payoff progression data structure
for epsilon_name in epsilon_values_research.keys():
    payoff_progression_data_research[epsilon_name] = {
        'rounds': list(range(1, n_research + 1)),
        'payoff_histories': []
    }

for sim in range(num_simulations):
    if sim % 20 == 0:
        print(f"Research simulation {sim}/{num_simulations}")

    for epsilon_name, epsilon_value in epsilon_values_research.items():
        # Reset model for each simulation
        research_model.reset()

        # Initialize EG algorithm with selected_papers as k
        eg = ExponentiatedGradientFI(k=selected_papers, epsilon=epsilon_value, n=n_research)

        # Create payoff generator that returns only alpha for selected papers
        def payoff_generator(round_num):
            alpha, selected_papers = research_model.generate_payoffs(round_num)
            # Return only alpha values for selected papers
            return np.array([alpha[paper_id] for paper_id in selected_papers])

        # Run algorithm
        regret_history, total_payoff, cumulative_payoffs, payoff_history = eg.run(payoff_generator)

        # Store payoff progression
        payoff_progression_data_research[epsilon_name]['payoff_histories'].append(payoff_history)

        # Store results
        research_results.append({
            'simulation': sim,
            'epsilon_type': epsilon_name,
            'epsilon_value': epsilon_value,
            'regret_history': regret_history,
            'total_payoff': total_payoff,
            'cumulative_regret': regret_history[-1] if len(regret_history) > 0 else 0
        })

print("\nResearch Payoffs simulation completed!")
print(f"Total results: {len(research_results)}")

# Calculate mean and std for payoff progression
for epsilon_name in epsilon_values_research.keys():
    payoff_histories = payoff_progression_data_research[epsilon_name]['payoff_histories']
    payoff_progression_data_research[epsilon_name]['mean_payoff'] = np.mean(payoff_histories, axis=0).tolist()
    payoff_progression_data_research[epsilon_name]['std_payoff'] = np.std(payoff_histories, axis=0).tolist()
    payoff_progression_data_research[epsilon_name]['epsilon_value'] = epsilon_values_research[epsilon_name]
    # Remove the raw histories to save memory
    del payoff_progression_data_research[epsilon_name]['payoff_histories']

# Prepare data for visualization (with payoff data)
for epsilon_name in epsilon_values_research.keys():
    # Get all results for this epsilon type
    epsilon_results = [r for r in research_results if r['epsilon_type'] == epsilon_name]
    all_regrets = [r['regret_history'] for r in epsilon_results]

    # Calculate mean and std for each round
    mean_regret = np.mean(all_regrets, axis=0)
    std_regret = np.std(all_regrets, axis=0)
    mean_payoff = payoff_progression_data_research[epsilon_name]['mean_payoff']
    std_payoff = payoff_progression_data_research[epsilon_name]['std_payoff']

    # Create data for each round
    for round_num in range(n_research):
        results_data_research.append({
            'round': round_num + 1,
            'epsilon_type': epsilon_name,
            'epsilon_value': epsilon_values_research[epsilon_name],
            'mean_regret': mean_regret[round_num],
            'std_regret': std_regret[round_num],
            'mean_payoff': mean_payoff[round_num],
            'std_payoff': std_payoff[round_num],
            'final_regret_mean': np.mean([r['cumulative_regret'] for r in epsilon_results]),
            'final_regret_std': np.std([r['cumulative_regret'] for r in epsilon_results]),
            'final_payoff_mean': np.mean([r['total_payoff'] for r in epsilon_results]),
            'final_payoff_std': np.std([r['total_payoff'] for r in epsilon_results])
        })

research_plot_df = pd.DataFrame(results_data_research)

# Save results to CSV
research_plot_df.to_csv(data_dir / 'research_payoffs_cluster_results.csv', index=False)
print(f"Results saved to: {data_dir / 'research_payoffs_cluster_results.csv'}")

# Statistical results
print("\nFinal regret statistics:")
for epsilon_name in epsilon_values_research.keys():
    subset = research_plot_df[research_plot_df['epsilon_type'] == epsilon_name]
    final_stats = subset.iloc[0]
    print(f"  {epsilon_name}: Mean = {final_stats['final_regret_mean']:.4f}, Std = {final_stats['final_regret_std']:.4f}")


# In[15]:


# Research Payoffs Visualization - Regret and Payoff Progression

import visualization

# Load the data
research_cluster_df = pd.read_csv(data_dir / 'research_payoffs_cluster_results.csv')

print("Research Payoffs Cluster Results - Data Overview:")
print(f"Shape: {research_cluster_df.shape}")
print(f"Columns: {list(research_cluster_df.columns)}")
print(f"Epsilon types: {research_cluster_df['epsilon_type'].unique()}")
print()

# Use the plot_regret_comparison function from visualization.py
print("Generating regret comparison plot...")
visualization.plot_regret_comparison(
    research_cluster_df,
    'Research Payoffs Environment (EG Algorithm)',
    'research_cluster_regret_comparison.png'
)

# Plot payoff progression from CSV data
print("Generating payoff progression plot from CSV...")
visualization.plot_payoff_comparison(
    research_cluster_df,
    "Research Payoffs (EG Algorithm)",
    "research_cluster_payoff_progression.png"
)

# Summary statistics
print("\nResearch Payoffs Cluster - Summary Statistics:")
print("=" * 50)

for epsilon_name in research_cluster_df["epsilon_type"].unique():
    subset = research_cluster_df[research_cluster_df["epsilon_type"] == epsilon_name]
    final_regrets = subset['final_regret_mean'].iloc[0]
    final_regret_std = subset['final_regret_std'].iloc[0]
    total_payoffs = subset['final_payoff_mean'].iloc[0]
    total_payoff_std = subset['final_payoff_std'].iloc[0]

    print(f"{epsilon_name} (ε={subset['epsilon_value'].iloc[0]:.6f}):")
    print(f"  Final Regret: {final_regrets:.4f} ± {final_regret_std:.4f}")
    print(f"  Total Payoff: {total_payoffs:.4f} ± {total_payoff_std:.4f}")
    print()

# Weight evolution analysis (for a single simulation)
print("Weight Evolution Analysis:")
print("=" * 30)

# Run a single simulation to analyze weight evolution
num_clusters_test = 4
papers_per_cluster_test = 8
selected_papers_test = num_clusters_test * 3  # Fixed: 3 papers per cluster
n_research_test = 1000

# Test with optimal epsilon (using selected_papers)
epsilon_optimal = np.sqrt(2 * np.log(selected_papers_test) / n_research_test)
research_model_test = ResearchPayoffs(
    num_clusters=num_clusters_test,
    papers_per_cluster=papers_per_cluster_test,
    regime_switch_prob=0.3,
    seed=42
)
eg_test = ExponentiatedGradientFI(k=selected_papers_test, epsilon=epsilon_optimal, n=n_research_test)

# Create payoff generator for EG
def payoff_generator_test(round_num):
    alpha, selected_papers = research_model_test.generate_payoffs(round_num)
    return np.array([alpha[paper_id] for paper_id in selected_papers])

# Run algorithm
regret_history, total_payoff, cumulative_payoffs, payoff_history = eg_test.run(payoff_generator_test)
weight_history = eg_test.get_weight_history()

# Analyze final weights
final_weights = weight_history[-1, :]
print("Final weight distribution (all selected papers):")
for i, weight in enumerate(final_weights):
    # Get cluster information for this paper index
    cluster_mapping = research_model_test.get_cluster_mapping()
    cluster_id = cluster_mapping[i]
    classification = research_model_test.get_paper_classification(i)
    print(f"  Paper {i} (Cluster {cluster_id}, {classification}): {weight:.4f}")

print(f"\nTotal payoff: {total_payoff:.4f}")
print(f"Final regret: {regret_history[-1]:.4f}")

# Show cluster information
print("\nCluster Information:")
cluster_info = research_model_test.get_cluster_info_for_researcher()
print(f"  Current regime: {cluster_info['current_regime']}")
print(f"  High cluster: {cluster_info['high_cluster']}")
print(f"  Middle cluster: {cluster_info['middle_cluster']}")
print(f"  Low clusters: {cluster_info['low_clusters']}")
print(f"  Cluster mapping: {cluster_info['cluster_mapping']}")

