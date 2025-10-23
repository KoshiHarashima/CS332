# Detailed ROI Time Series Plot for Pachinko Data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

def create_detailed_roi_plots():
    """Create detailed ROI time series plots"""
    
    # Set up paths
    current_dir = Path(__file__).parent
    data_dir = current_dir.parent / 'data'
    output_dir = current_dir.parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    data_file = data_dir / 'espace_5stores_daily_2024_2025.csv'
    data = pd.read_csv(data_file)
    
    # Normalize store names
    data['store'] = data['store'].str.replace('\ufeff', '')
    
    # Calculate ROI
    base_investment = 1000
    data['roi'] = data['avg_diff'] / base_investment
    
    # Set style
    plt.style.use('seaborn-v0_8')
    
    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(20, 15))
    fig.suptitle('Pachinko ROI Analysis - Detailed Time Series', fontsize=16, fontweight='bold')
    
    # Plot 1: Individual store ROI time series
    ax1 = axes[0, 0]
    stores = data['store'].unique()
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, store in enumerate(stores):
        store_data = data[data['store'] == store]
        ax1.plot(store_data.index, store_data['roi'], 
                label=store, color=colors[i], alpha=0.7, linewidth=1.5)
    
    ax1.set_title('ROI Time Series by Store', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Day')
    ax1.set_ylabel('ROI')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Plot 2: Rolling average ROI
    ax2 = axes[0, 1]
    window = 30  # 30-day rolling average
    
    for i, store in enumerate(stores):
        store_data = data[data['store'] == store]
        rolling_avg = store_data['roi'].rolling(window=window).mean()
        ax2.plot(store_data.index, rolling_avg, 
                label=f'{store} (30-day avg)', color=colors[i], linewidth=2)
    
    ax2.set_title(f'Rolling Average ROI (30-day window)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Day')
    ax2.set_ylabel('ROI')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Plot 3: ROI distribution by store
    ax3 = axes[1, 0]
    store_roi_data = [data[data['store'] == store]['roi'].values for store in stores]
    box_plot = ax3.boxplot(store_roi_data, labels=stores, patch_artist=True)
    
    # Color the boxes
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax3.set_title('ROI Distribution by Store', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Store')
    ax3.set_ylabel('ROI')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Plot 4: Cumulative ROI
    ax4 = axes[1, 1]
    
    for i, store in enumerate(stores):
        store_data = data[data['store'] == store]
        cumulative_roi = store_data['roi'].cumsum()
        ax4.plot(store_data.index, cumulative_roi, 
                label=store, color=colors[i], linewidth=2)
    
    ax4.set_title('Cumulative ROI by Store', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Day')
    ax4.set_ylabel('Cumulative ROI')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Plot 5: ROI volatility over time
    ax5 = axes[2, 0]
    window = 30  # 30-day rolling volatility
    
    for i, store in enumerate(stores):
        store_data = data[data['store'] == store]
        rolling_std = store_data['roi'].rolling(window=window).std()
        ax5.plot(store_data.index, rolling_std, 
                label=store, color=colors[i], linewidth=2)
    
    ax5.set_title(f'ROI Volatility (30-day rolling std)', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Day')
    ax5.set_ylabel('ROI Standard Deviation')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Store performance comparison
    ax6 = axes[2, 1]
    
    # Calculate performance metrics
    store_metrics = []
    for store in stores:
        store_data = data[data['store'] == store]
        metrics = {
            'store': store,
            'mean_roi': store_data['roi'].mean(),
            'std_roi': store_data['roi'].std(),
            'min_roi': store_data['roi'].min(),
            'max_roi': store_data['roi'].max(),
            'positive_days': (store_data['roi'] > 0).sum(),
            'total_days': len(store_data)
        }
        store_metrics.append(metrics)
    
    # Create performance comparison
    metrics_df = pd.DataFrame(store_metrics)
    
    # Plot mean ROI vs volatility
    scatter = ax6.scatter(metrics_df['std_roi'], metrics_df['mean_roi'], 
                         c=colors, s=100, alpha=0.7)
    
    # Add store labels
    for i, store in enumerate(stores):
        ax6.annotate(store, (metrics_df.iloc[i]['std_roi'], metrics_df.iloc[i]['mean_roi']),
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    ax6.set_title('Store Performance: Mean ROI vs Volatility', fontsize=14, fontweight='bold')
    ax6.set_xlabel('ROI Standard Deviation (Volatility)')
    ax6.set_ylabel('Mean ROI')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_file = output_dir / 'pachinko_detailed_roi_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Detailed ROI plot saved to: {output_file}")
    
    # Print summary statistics
    print("\n=== Detailed ROI Analysis ===")
    print(f"Total days: {len(data)}")
    print(f"Number of stores: {len(stores)}")
    print(f"Overall ROI range: [{data['roi'].min():.4f}, {data['roi'].max():.4f}]")
    print(f"Overall ROI mean: {data['roi'].mean():.4f}")
    print(f"Overall ROI std: {data['roi'].std():.4f}")
    
    print("\n=== Store Performance Ranking ===")
    metrics_df_sorted = metrics_df.sort_values('mean_roi', ascending=False)
    for i, (_, row) in enumerate(metrics_df_sorted.iterrows()):
        print(f"{i+1}. {row['store']}: Mean ROI = {row['mean_roi']:.4f}, "
              f"Volatility = {row['std_roi']:.4f}, "
              f"Positive days = {row['positive_days']}/{row['total_days']}")
    
    plt.show()

if __name__ == "__main__":
    create_detailed_roi_plots()
