# Here, we consider the Pachinko Payoffs based on ROI (Return on Investment).

# Shops open the data about the number of balls in and out each day.
# We calculate ROI as (balls_out - balls_in) / investment_amount, then normalize it to [0,1] range.
# So, instead of applying the EW algorithm to each gumbuling machines, 
# we apply the EW algorithm to each store, and the payoff is the normalized ROI for each day.

# each round, we generate the ROI-based payoff for each store.
# there's five stores.
# each day, the player chooses one store to play gumbuling machines.
# then the player gets ROI for each store (extracted from the data) and normalize it to [0,1]


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class EspacePayoffs:
    """PayoffGenerator using daily data from 5 Espace stores (ROI-based)"""
    
    def __init__(self, k):
        """
        Initialize ROI-based payoff generator by loading data from 5 Espace stores
        
        Args:
            k: Number of stores (for compatibility with A_afp.py)
        """
        # Set data file path
        current_dir = Path(__file__).parent
        data_dir = current_dir.parent / 'data'
        data_file_path = data_dir / 'espace_5stores_daily_2024_2025.csv'
        
        # Load data
        self.data = pd.read_csv(data_file_path)
        
        # Normalize store names (remove BOM characters)
        self.data['store'] = self.data['store'].str.replace('\ufeff', '')
        
        # Get store list
        self.stores = self.data['store'].unique()
        self.k = len(self.stores)  # Number of stores
        
        # Calculate avg_diff statistics
        self.min_avg_diff = self.data['avg_diff'].min()
        self.max_avg_diff = self.data['avg_diff'].max()
        
        # Pre-calculate ROI statistics (assuming investment amount of 1000)
        self.base_investment = 1000
        all_rois = self.data['avg_diff'] / self.base_investment
        self.min_roi = all_rois.min()
        self.max_roi = all_rois.max()
        
        # Pre-calculate average avg_diff for each store
        self.store_means = {}
        for store in self.stores:
            store_data = self.data[self.data['store'] == store]
            self.store_means[store] = store_data['avg_diff'].mean()
        
        # State variables
        self.current_day = 0
        self.total_days = len(self.data)

        for store, mean_val in self.store_means.items():
            print(f"  {store}: {mean_val:.2f}")
        
    def calculate_roi(self, avg_diff, base_investment=1000):
        """Calculate ROI (Return on Investment) from avg_diff"""
        # Treat avg_diff as profit, ROI = profit / investment amount
        # Negative values are treated as losses
        roi = avg_diff / base_investment
        return roi
    
    def normalize_roi(self, roi):
        """Normalize ROI to [0,1] range"""
        # Use pre-calculated ROI range
        normalized_roi = (roi - self.min_roi) / (self.max_roi - self.min_roi)
        return max(0, min(1, normalized_roi))  # Clip to 0-1 range
    
    def generate_payoffs(self, round_num):
        """Generate payoffs for each store in the specified round (ROI-based)"""
        if self.current_day >= self.total_days:
            # If data is exhausted, repeat the last day
            day_data = self.data.iloc[-1]
        else:
            day_data = self.data.iloc[self.current_day]
        
        # Calculate payoffs for each store (ROI-based)
        payoffs = np.zeros(self.k)
        for i, store in enumerate(self.stores):
            if day_data['store'] == store:
                # Use actual avg_diff for that day's store
                avg_diff = day_data['avg_diff']
            else:
                # For other stores, use their historical average
                avg_diff = self.store_means[store]
            
            # Calculate and normalize ROI
            roi = self.calculate_roi(avg_diff)
            normalized_payoff = self.normalize_roi(roi)
            payoffs[i] = normalized_payoff
        
        self.current_day += 1
        return payoffs
    
    def reset(self):
        """Reset state"""
        self.current_day = 0




