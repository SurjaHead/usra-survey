#!/usr/bin/env python3
"""
Simple MLP timing plotter - assumes data is already downloaded locally.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_mlp_data(data_dir="mlp_data"):
    """Load MLP timing data from local directory"""
    all_data = []
    
    # Find all activation_timings.csv files
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file == "activation_timings.csv":
                file_path = os.path.join(root, file)
                
                try:
                    # Load CSV
                    df = pd.read_csv(file_path)
                    
                    # Extract activation name from path
                    # Format: mlp_data/MLP_ReLU_20250101_123456/activation_timings.csv
                    parts = file_path.split(os.sep)
                    for part in parts:
                        if part.startswith("MLP_"):
                            activation = part.split("_")[1]
                            break
                    else:
                        activation = "Unknown"
                    
                    df['activation'] = activation
                    print(f"✅ Loaded {activation}: {len(df)} epochs")
                    all_data.append(df)
                    
                except Exception as e:
                    print(f"❌ Error loading {file_path}: {e}")
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return None

def plot_timing_data(df):
    """Create timing plot matching the attached image style"""
    
    # Set up plot
    plt.figure(figsize=(12, 8))
    
    # Colors for each activation
    colors = {
        'ReLU': '#ff7f0e',      # Orange
        'GELU': '#1f77b4',      # Blue
        'Sigmoid': '#2ca02c',   # Green  
        'Tanh': '#d62728',      # Red
        'ELU': '#9467bd',       # Purple
        'LeakyReLU': '#8c564b',  # Brown
        'SiLU': '#17becf',      # Cyan
        'Mish': '#bcbd22'       # Olive
    }
    
    # Plot each activation
    for activation in sorted(df['activation'].unique()):
        data = df[df['activation'] == activation].sort_values('epoch')
        
        plt.plot(data['epoch'], data['total_time'], 
                label=activation, 
                linewidth=2.5, 
                color=colors.get(activation, 'gray'),
                alpha=0.8)
        
        # Add final point
        if len(data) > 0:
            plt.scatter(data['epoch'].iloc[-1], data['total_time'].iloc[-1], 
                       color=colors.get(activation, 'gray'), s=80, zorder=5)
    
    # Styling to match the image
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Total Activation Time (ms)', fontsize=14)
    plt.title('MLP Activation Function Timing Comparison', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Set reasonable limits
    plt.xlim(0, df['epoch'].max())
    plt.ylim(df['total_time'].min() * 0.95, df['total_time'].max() * 1.05)
    
    plt.tight_layout()
    plt.savefig('mlp_timing_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Try to load data
    df = load_mlp_data("mlp_data")  # Change this path as needed
    
    if df is None:
        print("❌ No data found. Make sure to download from Modal first:")
        print("   modal volume get mlp-csv-logs mlp_data/")
        return
    
    print(f"📊 Found data for {len(df['activation'].unique())} activations")
    
    # Print summary
    for activation in sorted(df['activation'].unique()):
        data = df[df['activation'] == activation]
        final_time = data['total_time'].iloc[-1] if len(data) > 0 else 0
        print(f"{activation}: {final_time:.4f} ms")
    
    # Create plot
    plot_timing_data(df)

if __name__ == "__main__":
    main()
