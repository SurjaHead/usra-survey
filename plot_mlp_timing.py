#!/usr/bin/env python3
"""
Plot MLP activation timing data from Modal storage.
Downloads CSV logs and creates timing plots similar to the attached image.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import subprocess
import sys

def download_modal_data(volume_name="mlp-csv-logs", local_dir="mlp_data"):
    """Download MLP data from Modal storage"""
    try:
        print(f"📥 Downloading data from Modal volume: {volume_name}")
        
        # Create local directory
        os.makedirs(local_dir, exist_ok=True)
        
        # Download from Modal
        result = subprocess.run([
            "modal", "volume", "get", volume_name, local_dir
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Data downloaded to {local_dir}/")
            return local_dir
        else:
            print(f"❌ Failed to download: {result.stderr}")
            return None
            
    except FileNotFoundError:
        print("❌ Modal CLI not found. Install with: pip install modal")
        return None

def find_activation_timing_files(data_dir):
    """Find all activation_timings.csv files in the downloaded data"""
    timing_files = []
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file == "activation_timings.csv":
                timing_files.append(os.path.join(root, file))
    
    return timing_files

def extract_activation_name(file_path):
    """Extract activation function name from file path"""
    # Path format: mlp_data/MLP_ReLU_20250101_123456/activation_timings.csv
    parts = file_path.split(os.sep)
    for part in parts:
        if part.startswith("MLP_"):
            activation = part.split("_")[1]  # Extract activation name
            return activation
    return "Unknown"

def load_and_process_data(timing_files):
    """Load all timing data and combine into single DataFrame"""
    all_data = []
    
    for file_path in timing_files:
        try:
            # Load CSV
            df = pd.read_csv(file_path)
            
            # Extract activation name
            activation = extract_activation_name(file_path)
            
            # Add activation column
            df['activation'] = activation
            
            # Calculate total time if not present
            if 'total_time' not in df.columns:
                # Sum all layer columns (excluding epoch and activation)
                layer_cols = [col for col in df.columns if col.startswith('layer_')]
                df['total_time'] = df[layer_cols].sum(axis=1)
            
            all_data.append(df)
            print(f"✅ Loaded {activation}: {len(df)} epochs")
            
        except Exception as e:
            print(f"❌ Error loading {file_path}: {e}")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        return combined_df
    else:
        return None

def create_timing_plot(df, save_path="mlp_activation_timing.png"):
    """Create the timing plot similar to the attached image"""
    
    # Set up the plot style
    plt.style.use('default')
    sns.set_palette("tab10")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define colors for consistency
    colors = {
        'ReLU': '#1f77b4',      # Blue
        'GELU': '#ff7f0e',      # Orange  
        'Sigmoid': '#2ca02c',   # Green
        'Tanh': '#d62728',      # Red
        'ELU': '#9467bd',       # Purple
        'LeakyReLU': '#8c564b',  # Brown
        'SiLU': '#e377c2',      # Pink
        'Mish': '#7f7f7f'       # Gray
    }
    
    # Plot each activation function
    activations = df['activation'].unique()
    
    for activation in sorted(activations):
        activation_data = df[df['activation'] == activation].copy()
        activation_data = activation_data.sort_values('epoch')
        
        color = colors.get(activation, None)
        
        # Plot with some transparency and smoothing
        ax.plot(activation_data['epoch'], activation_data['total_time'], 
               label=activation, linewidth=2.5, alpha=0.8, color=color)
        
        # Add final point marker
        if len(activation_data) > 0:
            final_epoch = activation_data['epoch'].iloc[-1]
            final_time = activation_data['total_time'].iloc[-1]
            ax.scatter(final_epoch, final_time, color=color, s=80, alpha=0.9, zorder=5)
    
    # Customize the plot to match the image
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Total Activation Time (ms)', fontsize=14)
    ax.set_title('MLP Activation Function Timing Comparison', fontsize=16, fontweight='bold')
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    
    # Set axis limits
    if len(df) > 0:
        ax.set_xlim(0, df['epoch'].max())
        ax.set_ylim(df['total_time'].min() * 0.95, df['total_time'].max() * 1.05)
    
    # Tight layout
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 Plot saved to {save_path}")
    
    # Show plot
    plt.show()
    
    return fig, ax

def print_summary_stats(df):
    """Print summary statistics"""
    print("\n📊 Summary Statistics:")
    print("=" * 50)
    
    for activation in sorted(df['activation'].unique()):
        activation_data = df[df['activation'] == activation]
        
        if len(activation_data) > 0:
            final_time = activation_data['total_time'].iloc[-1]
            avg_time = activation_data['total_time'].mean()
            min_time = activation_data['total_time'].min()
            max_time = activation_data['total_time'].max()
            
            print(f"{activation:>10}: Final={final_time:.4f}ms, "
                  f"Avg={avg_time:.4f}ms, Min={min_time:.4f}ms, Max={max_time:.4f}ms")

def main():
    """Main function to download data and create plots"""
    print("🚀 MLP Activation Timing Plotter")
    print("=" * 40)
    
    # Download data from Modal
    data_dir = download_modal_data()
    if not data_dir:
        print("❌ Failed to download data. Exiting.")
        return
    
    # Find timing files
    timing_files = find_activation_timing_files(data_dir)
    
    if not timing_files:
        print(f"❌ No activation_timings.csv files found in {data_dir}")
        return
    
    print(f"📁 Found {len(timing_files)} timing files")
    
    # Load and process data
    df = load_and_process_data(timing_files)
    
    if df is None or len(df) == 0:
        print("❌ No data loaded. Exiting.")
        return
    
    print(f"📊 Loaded data for {len(df['activation'].unique())} activation functions")
    
    # Print summary stats
    print_summary_stats(df)
    
    # Create plot
    create_timing_plot(df)
    
    print("✅ Complete!")

if __name__ == "__main__":
    main()
