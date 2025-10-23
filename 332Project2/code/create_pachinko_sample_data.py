# Create sample Pachinko data for ROI time series plotting

import pandas as pd
import numpy as np
from pathlib import Path

def create_sample_pachinko_data():
    """Create sample Pachinko data for demonstration"""
    
    # Set up paths
    current_dir = Path(__file__).parent
    data_dir = current_dir.parent / 'data'
    data_dir.mkdir(exist_ok=True)
    
    # Parameters
    num_days = 365  # One year of data
    stores = ['Store_A', 'Store_B', 'Store_C', 'Store_D', 'Store_E']
    
    # Create sample data
    data = []
    
    for day in range(num_days):
        for store in stores:
            # Generate realistic ROI data with different characteristics per store
            if store == 'Store_A':
                # High performing store
                base_roi = 0.15
                volatility = 0.05
            elif store == 'Store_B':
                # Medium performing store
                base_roi = 0.08
                volatility = 0.03
            elif store == 'Store_C':
                # Low performing store
                base_roi = 0.02
                volatility = 0.02
            elif store == 'Store_D':
                # Volatile store
                base_roi = 0.05
                volatility = 0.08
            else:  # Store_E
                # Declining store
                base_roi = 0.10 - (day / num_days) * 0.08  # Declining over time
                volatility = 0.04
            
            # Add seasonal effects
            seasonal_effect = 0.02 * np.sin(2 * np.pi * day / 365)
            
            # Add random noise
            noise = np.random.normal(0, volatility)
            
            # Calculate final ROI
            roi = base_roi + seasonal_effect + noise
            
            # Calculate avg_diff (assuming base investment of 1000)
            base_investment = 1000
            avg_diff = roi * base_investment
            
            data.append({
                'day': day,
                'store': store,
                'avg_diff': avg_diff,
                'roi': roi
            })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_file = data_dir / 'espace_5stores_daily_2024_2025.csv'
    df.to_csv(output_file, index=False)
    
    print(f"Sample Pachinko data created: {output_file}")
    print(f"Data shape: {df.shape}")
    print(f"Stores: {df['store'].unique()}")
    print(f"ROI range: [{df['roi'].min():.4f}, {df['roi'].max():.4f}]")
    
    return df

if __name__ == "__main__":
    create_sample_pachinko_data()
