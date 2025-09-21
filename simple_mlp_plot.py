#!/usr/bin/env python3
"""
Simple MLP timing plotter using only matplotlib and built-in libraries.
"""

import os
import csv
import matplotlib.pyplot as plt

def load_mlp_data(data_dir="mlp_data"):
    """Load MLP timing data from CSV files"""
    all_data = {}
    
    # Walk through directory structure
    for root, dirs, files in os.walk(data_dir):
        if "activation_timings.csv" in files:
            file_path = os.path.join(root, "activation_timings.csv")
            
            # Extract activation name from directory name
            dir_name = os.path.basename(root)
            if "_" in dir_name:
                parts = dir_name.split("_")
                if len(parts) >= 2:
                    activation = parts[1]  # MLP_ReLU_timestamp -> ReLU
                else:
                    activation = "Unknown"
            else:
                activation = "Unknown"
            
            try:
                epochs = []
                total_times = []
                
                with open(file_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        epochs.append(int(row['epoch']))
                        total_times.append(float(row['total_time']))
                
                all_data[activation] = {
                    'epochs': epochs,
                    'total_times': total_times
                }
                print(f"✅ Loaded {activation}: {len(epochs)} epochs, final time: {total_times[-1]:.4f}ms")
                
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
    
    return all_data

def create_plot(data):
    """Create timing plot matching the attached image style"""
    
    plt.figure(figsize=(12, 8))
    
    # Colors for each activation (matching your image)
    colors = {
        'ReLU': '#ff7f0e',      # Orange
        'GELU': '#1f77b4',      # Blue
        'Sigmoid': '#2ca02c',   # Green  
        'Tanh': '#d62728',      # Red
        'ELU': '#9467bd',       # Purple
        'LeakyReLU': '#8c564b', # Brown
        'SiLU': '#17becf',      # Cyan
        'Mish': '#7f7f7f'       # Gray
    }
    
    # Plot each activation
    for activation, values in sorted(data.items()):
        epochs = values['epochs']
        times = values['total_times']
        
        color = colors.get(activation, 'black')
        
        plt.plot(epochs, times, 
                label=activation, 
                linewidth=2.5, 
                color=color,
                alpha=0.85)
        
        # Add final point marker
        if epochs and times:
            plt.scatter(epochs[-1], times[-1], 
                       color=color, s=100, alpha=0.9, zorder=5,
                       edgecolors='white', linewidth=1)
    
    # Styling to match your image
    plt.xlabel('Epoch', fontsize=14, fontweight='bold')
    plt.ylabel('Total Activation Time (ms)', fontsize=14, fontweight='bold')
    plt.title('MLP Activation Function Performance', fontsize=16, fontweight='bold', pad=20)
    
    # Grid
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    
    # Legend
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12,
              frameon=True, fancybox=True, shadow=True)
    
    # Clean up spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig('mlp_activation_timing.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig('mlp_activation_timing.pdf', bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print("📊 Plot saved as mlp_activation_timing.png and .pdf")
    plt.show()

def print_summary(data):
    """Print summary statistics"""
    print("\n" + "="*60)
    print("📊 MLP ACTIVATION FUNCTION PERFORMANCE SUMMARY")
    print("="*60)
    
    print(f"{'Activation':<12} {'Final (ms)':<12} {'Min (ms)':<10} {'Max (ms)':<10} {'Epochs':<8}")
    print("-" * 60)
    
    for activation, values in sorted(data.items()):
        times = values['total_times']
        if times:
            final_time = times[-1]
            min_time = min(times)
            max_time = max(times)
            epochs = len(times)
            
            print(f"{activation:<12} {final_time:<12.4f} {min_time:<10.4f} {max_time:<10.4f} {epochs:<8}")

def main():
    print("🚀 Simple MLP Activation Timing Plotter")
    print("=" * 50)
    
    # Load data
    data = load_mlp_data("mlp_data")
    
    if not data:
        print("❌ No data found. Make sure mlp_data/ directory exists with CSV files.")
        return
    
    print(f"\n✅ Successfully loaded data for {len(data)} activation functions")
    
    # Print summary
    print_summary(data)
    
    # Create plot
    print("\n🎨 Creating visualization...")
    create_plot(data)
    
    print("\n✅ Analysis complete!")
    print("📁 Generated files:")
    print("   - mlp_activation_timing.png")
    print("   - mlp_activation_timing.pdf")

if __name__ == "__main__":
    main()
