# Pachinko ROI Time Series Plot
# This script creates time series plots for ROI data from Pachinko stores

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

def create_pachinko_roi_plot(data_file_path=None, output_dir=None):
    """
    Create ROI time series plots for Pachinko data
    
    Args:
        data_file_path: Path to the Pachinko data CSV file
        output_dir: Directory to save the plots
    """
    
    # Set default paths if not provided
    if data_file_path is None:
        current_dir = Path(__file__).parent
        data_file_path = current_dir.parent / 'data' / 'espace_5stores_daily_2024_2025.csv'
    
    if output_dir is None:
        current_dir = Path(__file__).parent
        output_dir = current_dir.parent / 'figures'
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Load data
        print(f"Loading data from: {data_file_path}")
        data = pd.read_csv(data_file_path)
        
        # Normalize store names (remove BOM characters)
        data['store'] = data['store'].str.replace('\ufeff', '')
        
        # Get store list
        stores = data['store'].unique()
        print(f"Found {len(stores)} stores: {stores}")
        
        # Calculate ROI
        base_investment = 1000
        data['roi'] = data['avg_diff'] / base_investment
        
        # Create time series plot
        plt.figure(figsize=(15, 10))
        
        # Plot 1: ROI time series for each store
        plt.subplot(2, 2, 1)
        for store in stores:
            store_data = data[data['store'] == store]
            plt.plot(store_data.index, store_data['roi'], label=store, alpha=0.7)
        
        plt.title('ROI Time Series by Store')
        plt.xlabel('Day')
        plt.ylabel('ROI')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Average ROI by store
        plt.subplot(2, 2, 2)
        store_avg_roi = data.groupby('store')['roi'].mean().sort_values(ascending=False)
        bars = plt.bar(range(len(store_avg_roi)), store_avg_roi.values)
        plt.title('Average ROI by Store')
        plt.xlabel('Store')
        plt.ylabel('Average ROI')
        plt.xticks(range(len(store_avg_roi)), store_avg_roi.index, rotation=45)
        
        # Color bars based on performance
        for i, bar in enumerate(bars):
            if store_avg_roi.iloc[i] > 0:
                bar.set_color('green')
            else:
                bar.set_color('red')
        
        # Plot 3: ROI distribution
        plt.subplot(2, 2, 3)
        plt.hist(data['roi'], bins=30, alpha=0.7, edgecolor='black')
        plt.title('ROI Distribution')
        plt.xlabel('ROI')
        plt.ylabel('Frequency')
        plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Break-even')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: ROI volatility by store
        plt.subplot(2, 2, 4)
        store_volatility = data.groupby('store')['roi'].std().sort_values(ascending=False)
        bars = plt.bar(range(len(store_volatility)), store_volatility.values)
        plt.title('ROI Volatility by Store (Standard Deviation)')
        plt.xlabel('Store')
        plt.ylabel('ROI Standard Deviation')
        plt.xticks(range(len(store_volatility)), store_volatility.index, rotation=45)
        
        # Color bars based on volatility
        for i, bar in enumerate(bars):
            if store_volatility.iloc[i] > store_volatility.median():
                bar.set_color('red')
            else:
                bar.set_color('blue')
        
        plt.tight_layout()
        
        # Save the plot
        output_file = output_dir / 'pachinko_roi_analysis.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_file}")
        
        # Show summary statistics
        print("\n=== ROI Summary Statistics ===")
        print(f"Total days: {len(data)}")
        print(f"Number of stores: {len(stores)}")
        print(f"Overall ROI range: [{data['roi'].min():.4f}, {data['roi'].max():.4f}]")
        print(f"Overall ROI mean: {data['roi'].mean():.4f}")
        print(f"Overall ROI std: {data['roi'].std():.4f}")
        
        print("\n=== Store-wise Statistics ===")
        for store in stores:
            store_data = data[data['store'] == store]
            print(f"{store}:")
            print(f"  Days: {len(store_data)}")
            print(f"  ROI range: [{store_data['roi'].min():.4f}, {store_data['roi'].max():.4f}]")
            print(f"  ROI mean: {store_data['roi'].mean():.4f}")
            print(f"  ROI std: {store_data['roi'].std():.4f}")
        
        plt.show()
        
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_file_path}")
        print("Please check the file path and ensure the data file exists.")
    except Exception as e:
        print(f"Error creating plot: {e}")

if __name__ == "__main__":
    create_pachinko_roi_plot()
