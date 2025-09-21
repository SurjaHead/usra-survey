#!/usr/bin/env python3
"""
Comprehensive Neural Network Analysis Script
Analyzes activation timing and validation accuracy for all models:
- MLP (Multi-Layer Perceptron)
- CNN (ResNet-18) 
- GNN (Graph Convolutional Network)
- Transformer (BERT)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess
from pathlib import Path

# Set much larger font sizes globally
plt.rcParams.update({
    'font.size': 22,
    'axes.titlesize': 28,
    'axes.labelsize': 24,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 16,  # Keep legend smaller to avoid covering lines
    'figure.titlesize': 32
})

def setup_plotting_style():
    """Set up consistent plotting style"""
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'

def download_modal_data(volume_name, local_dir):
    """Download data from Modal volume"""
    try:
        print(f"📥 Downloading {volume_name} data from Modal...")
        result = subprocess.run([
            'modal', 'volume', 'get', volume_name, '/', local_dir
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print(f"✅ Successfully downloaded {volume_name} data to {local_dir}/")
            return True
        else:
            print(f"❌ Download failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("⏰ Download timed out")
        return False
    except FileNotFoundError:
        print("❌ Modal CLI not found. Install with: pip install modal")
        return False

# ============================================================================
# MLP ANALYSIS
# ============================================================================

def load_mlp_data(data_dir="mlp_data"):
    """Load MLP timing and training data"""
    if not os.path.exists(data_dir):
        print(f"📁 MLP data directory {data_dir}/ not found.")
        if not download_modal_data("mlp-csv-logs", data_dir):
            return None, None
    
    print(f"📁 Loading MLP data from {data_dir}/")
    
    all_timing_data = []
    all_training_data = []
    
    for root, dirs, files in os.walk(data_dir):
        if "activation_timings.csv" in files:
            timing_file = os.path.join(root, "activation_timings.csv")
            training_file = os.path.join(root, "training_metrics.csv")
            
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
                # Load timing data
                timing_df = pd.read_csv(timing_file)
                timing_df['activation'] = activation
                all_timing_data.append(timing_df)
                
                # Load training data if available
                if os.path.exists(training_file):
                    training_df = pd.read_csv(training_file)
                    training_df['activation'] = activation
                    all_training_data.append(training_df)
                
                print(f"✅ Loaded MLP {activation}: {len(timing_df)} epochs")
            except Exception as e:
                print(f"❌ Error loading MLP {timing_file}: {e}")
    
    timing_combined = pd.concat(all_timing_data, ignore_index=True) if all_timing_data else None
    training_combined = pd.concat(all_training_data, ignore_index=True) if all_training_data else None
    
    return timing_combined, training_combined

def create_mlp_timing_plot(df):
    """Create MLP activation timing plot"""
    if df is None or len(df) == 0:
        return None, None
        
    setup_plotting_style()
    fig, ax = plt.subplots(figsize=(14, 10))
    
    colors = sns.color_palette("husl", len(df['activation'].unique()))
    
    for i, activation in enumerate(sorted(df['activation'].unique())):
        activation_data = df[df['activation'] == activation]
        ax.plot(activation_data['epoch'], activation_data['total_time'], 
                label=activation, color=colors[i], linewidth=3, alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=24, fontweight='bold')
    ax.set_ylabel('Activation Time (ms)', fontsize=24, fontweight='bold')
    ax.set_title('MLP Activation Timing by Function', fontsize=28, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=14, frameon=True, shadow=True)  # Smaller legend for MLP
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 50)  # Set x-axis limits from 0 to 50
    
    plt.tight_layout()
    plt.savefig('mlp_activation_timing.pdf', bbox_inches='tight', facecolor='white')
    
    return fig, ax

def create_mlp_validation_plot(df):
    """Create MLP validation accuracy plot"""
    if df is None or len(df) == 0:
        return None, None
        
    setup_plotting_style()
    fig, ax = plt.subplots(figsize=(14, 10))
    
    colors = sns.color_palette("husl", len(df['activation'].unique()))
    
    for i, activation in enumerate(sorted(df['activation'].unique())):
        activation_data = df[df['activation'] == activation]
        ax.plot(activation_data['epoch'], activation_data['val_acc'] * 100, 
                label=activation, color=colors[i], linewidth=3, alpha=0.8)mak
    
    ax.set_xlabel('Epoch', fontsize=24, fontweight='bold')
    ax.set_ylabel('Validation Accuracy (%)', fontsize=24, fontweight='bold')
    ax.set_title('MLP Validation Accuracy by Activation Function', fontsize=28, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=14, frameon=True, shadow=True)  # Smaller legend for MLP
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 50)  # Set x-axis limits from 0 to 50
    
    plt.tight_layout()
    plt.savefig('mlp_validation_accuracy.pdf', bbox_inches='tight', facecolor='white')
    
    return fig, ax

# ============================================================================
# CNN ANALYSIS
# ============================================================================

def load_cnn_data(data_dir="cnn_data"):
    """Load CNN timing and training data"""
    if not os.path.exists(data_dir):
        print(f"📁 CNN data directory {data_dir}/ not found.")
        if not download_modal_data("cnn-csv-logs", data_dir):
            return None, None
    
    print(f"📁 Loading CNN data from {data_dir}/")
    
    all_timing_data = []
    all_training_data = []
    
    for root, dirs, files in os.walk(data_dir):
        if "activation_timings.csv" in files:
            timing_file = os.path.join(root, "activation_timings.csv")
            training_file = os.path.join(root, "training_metrics.csv")
            
            dir_name = os.path.basename(root)
            if "_" in dir_name:
                parts = dir_name.split("_")
                if len(parts) >= 2:
                    activation = parts[1]  # ResNet18_ReLU_timestamp -> ReLU
                else:
                    activation = "Unknown"
            else:
                activation = "Unknown"
            
            try:
                # Load timing data
                timing_df = pd.read_csv(timing_file)
                timing_df['activation'] = activation
                all_timing_data.append(timing_df)
                
                # Load training data if available
                if os.path.exists(training_file):
                    training_df = pd.read_csv(training_file)
                    training_df['activation'] = activation
                    all_training_data.append(training_df)
                
                print(f"✅ Loaded CNN {activation}: {len(timing_df)} epochs")
            except Exception as e:
                print(f"❌ Error loading CNN {timing_file}: {e}")
    
    timing_combined = pd.concat(all_timing_data, ignore_index=True) if all_timing_data else None
    training_combined = pd.concat(all_training_data, ignore_index=True) if all_training_data else None
    
    return timing_combined, training_combined

def create_cnn_timing_plot(df):
    """Create CNN activation timing plot"""
    if df is None or len(df) == 0:
        return None, None
        
    setup_plotting_style()
    fig, ax = plt.subplots(figsize=(14, 10))
    
    colors = sns.color_palette("husl", len(df['activation'].unique()))
    
    for i, activation in enumerate(sorted(df['activation'].unique())):
        activation_data = df[df['activation'] == activation]
        ax.plot(activation_data['epoch'], activation_data['total_time'], 
                label=activation, color=colors[i], linewidth=3, alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=24, fontweight='bold')
    ax.set_ylabel('Activation Time (ms)', fontsize=24, fontweight='bold')
    ax.set_title('CNN (ResNet-18) Activation Timing by Function', fontsize=28, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=14, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cnn_activation_timing.pdf', bbox_inches='tight', facecolor='white')
    
    return fig, ax

def create_cnn_validation_plot(df):
    """Create CNN validation accuracy plot"""
    if df is None or len(df) == 0:
        return None, None
        
    setup_plotting_style()
    fig, ax = plt.subplots(figsize=(14, 10))
    
    colors = sns.color_palette("husl", len(df['activation'].unique()))
    
    for i, activation in enumerate(sorted(df['activation'].unique())):
        activation_data = df[df['activation'] == activation]
        ax.plot(activation_data['epoch'], activation_data['val_acc'] * 100, 
                label=activation, color=colors[i], linewidth=3, alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=24, fontweight='bold')
    ax.set_ylabel('Validation Accuracy (%)', fontsize=24, fontweight='bold')
    ax.set_title('CNN (ResNet-18) Validation Accuracy by Activation Function', fontsize=28, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=14, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cnn_validation_accuracy.pdf', bbox_inches='tight', facecolor='white')
    
    return fig, ax

# ============================================================================
# GNN ANALYSIS
# ============================================================================

def load_gnn_data(data_dir="gnn_data"):
    """Load GNN timing and training data"""
    if not os.path.exists(data_dir):
        print(f"📁 GNN data directory {data_dir}/ not found.")
        if not download_modal_data("gnn-csv-logs", data_dir):
            return None, None
    
    print(f"📁 Loading GNN data from {data_dir}/")
    
    timing_activation_data = {}
    training_activation_data = {}
    
    run_dirs = [d for d in os.listdir(data_dir) 
                if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("GNN_Cora_")]
    
    for run_dir in sorted(run_dirs):
        run_path = os.path.join(data_dir, run_dir)
        
        parts = run_dir.split('_')
        if len(parts) >= 3:
            activation = parts[2]  # GNN_Cora_ReLU_timestamp -> ReLU
        else:
            continue
        
        timing_file = os.path.join(run_path, "activation_timings.csv")
        training_file = os.path.join(run_path, "training_log.csv")
        
        if os.path.exists(timing_file):
            try:
                timing_df = pd.read_csv(timing_file)
                if not timing_df.empty and 'total_time' in timing_df.columns:
                    timing_data = timing_df['total_time'].values
                    timing_activation_data[activation] = timing_data
                    
                    # Load training data if available
                    if os.path.exists(training_file):
                        training_df = pd.read_csv(training_file)
                        if not training_df.empty and 'val_acc' in training_df.columns:
                            training_activation_data[activation] = training_df
                    
                    print(f"✅ Loaded GNN {activation}: {len(timing_data)} epochs")
            except Exception as e:
                print(f"❌ Error loading GNN {timing_file}: {e}")
    
    # Convert timing data to DataFrame
    timing_df = None
    if timing_activation_data:
        max_epochs = max(len(data) for data in timing_activation_data.values())
        timing_df_data = {}
        for activation, timing_data in timing_activation_data.items():
            padded_data = np.pad(timing_data, (0, max_epochs - len(timing_data)), 
                               constant_values=np.nan) if len(timing_data) < max_epochs else timing_data
            timing_df_data[activation] = padded_data
        timing_df = pd.DataFrame(timing_df_data)
        timing_df.index.name = 'epoch'
    
    # Convert training data to DataFrame
    training_df = None
    if training_activation_data:
        all_training_data = []
        for activation, data in training_activation_data.items():
            data_copy = data.copy()
            data_copy['activation'] = activation
            all_training_data.append(data_copy)
        training_df = pd.concat(all_training_data, ignore_index=True)
    
    return timing_df, training_df

def create_gnn_timing_plot(df):
    """Create GNN activation timing plot"""
    if df is None or len(df) == 0:
        return None, None
        
    setup_plotting_style()
    fig, ax = plt.subplots(figsize=(14, 10))
    
    colors = sns.color_palette("husl", len(df.columns))
    
    for i, activation in enumerate(sorted(df.columns)):
        activation_data = df[activation].dropna()
        epochs = range(len(activation_data))
        ax.plot(epochs, activation_data, label=activation, color=colors[i], linewidth=3, alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=24, fontweight='bold')
    ax.set_ylabel('Activation Time (ms)', fontsize=24, fontweight='bold')
    ax.set_title('GNN (Graph Convolutional Network) Activation Timing by Function', fontsize=28, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=14, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gnn_activation_timing.pdf', bbox_inches='tight', facecolor='white')
    
    return fig, ax

def create_gnn_validation_plot(df):
    """Create GNN validation accuracy plot"""
    if df is None or len(df) == 0:
        return None, None
        
    setup_plotting_style()
    fig, ax = plt.subplots(figsize=(14, 10))
    
    colors = sns.color_palette("husl", len(df['activation'].unique()))
    
    for i, activation in enumerate(sorted(df['activation'].unique())):
        activation_data = df[df['activation'] == activation].copy()
        activation_data = activation_data.sort_values('epoch')
        
        if 'val_acc' in activation_data.columns:
            ax.plot(activation_data['epoch'], activation_data['val_acc'] * 100, 
                    label=activation, color=colors[i], linewidth=3, alpha=0.8)
    
    ax.set_xlabel('Epoch', fontsize=24, fontweight='bold')
    ax.set_ylabel('Validation Accuracy (%)', fontsize=24, fontweight='bold')
    ax.set_title('GNN (Graph Convolutional Network) Validation Accuracy by Activation Function', fontsize=28, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=14, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gnn_validation_accuracy.pdf', bbox_inches='tight', facecolor='white')
    
    return fig, ax

# ============================================================================
# TRANSFORMER ANALYSIS
# ============================================================================

def remove_outliers(df, threshold_ms=1.5):
    """Remove outliers above threshold"""
    outliers = df[df['total_time'] > threshold_ms]
    df_filtered = df[df['total_time'] <= threshold_ms].copy()
    
    if len(outliers) > 0:
        print(f"🧹 Removed {len(outliers)} outliers above {threshold_ms}ms threshold")
    
    return df_filtered

def load_transformer_data(data_dir="transformer_data"):
    """Load Transformer timing and training data"""
    if not os.path.exists(data_dir):
        print(f"📁 Transformer data directory {data_dir}/ not found.")
        if not download_modal_data("transformer-csv-logs", data_dir):
            return None, None
    
    print(f"📁 Loading Transformer data from {data_dir}/")
    
    timing_activation_data = {}
    training_activation_data = {}
    
    run_dirs = [d for d in os.listdir(data_dir) 
                if os.path.isdir(os.path.join(data_dir, d)) and d.startswith("BERT_")]
    
    for run_dir in sorted(run_dirs):
        run_path = os.path.join(data_dir, run_dir)
        
        parts = run_dir.split('_')
        if len(parts) >= 2:
            activation = parts[1]  # BERT_ReLU_timestamp -> ReLU
        else:
            continue
        
        timing_file = os.path.join(run_path, "activation_timings.csv")
        training_file = os.path.join(run_path, "training_log.csv")
        
        if os.path.exists(timing_file):
            try:
                timing_df = pd.read_csv(timing_file)
                if not timing_df.empty and 'total_time' in timing_df.columns and 'step' in timing_df.columns:
                    timing_data = timing_df[['step', 'total_time']].copy()
                    timing_activation_data[activation] = timing_data
                    
                    # Load training data if available
                    if os.path.exists(training_file):
                        training_df = pd.read_csv(training_file)
                        if not training_df.empty and 'val_acc' in training_df.columns:
                            training_activation_data[activation] = training_df
                    
                    print(f"✅ Loaded Transformer {activation}: {len(timing_data)} steps")
            except Exception as e:
                print(f"❌ Error loading Transformer {timing_file}: {e}")
    
    # Convert timing data to DataFrame
    timing_df = None
    if timing_activation_data:
        timing_all_data = []
        for activation, data in timing_activation_data.items():
            data_copy = data.copy()
            data_copy['activation'] = activation
            timing_all_data.append(data_copy)
        timing_df = pd.concat(timing_all_data, ignore_index=True)
        timing_df = remove_outliers(timing_df, threshold_ms=1.5)
    
    # Convert training data to DataFrame
    training_df = None
    if training_activation_data:
        training_all_data = []
        for activation, data in training_activation_data.items():
            data_copy = data.copy()
            data_copy['activation'] = activation
            training_all_data.append(data_copy)
        training_df = pd.concat(training_all_data, ignore_index=True)
    
    return timing_df, training_df

def create_transformer_timing_plot(df):
    """Create Transformer activation timing plot"""
    if df is None or len(df) == 0:
        return None, None
        
    setup_plotting_style()
    fig, ax = plt.subplots(figsize=(14, 10))
    
    colors = sns.color_palette("husl", len(df['activation'].unique()))
    
    for i, activation in enumerate(sorted(df['activation'].unique())):
        activation_data = df[df['activation'] == activation]
        ax.plot(activation_data['step'], activation_data['total_time'], 
                label=activation, color=colors[i], linewidth=3, alpha=0.8)
    
    ax.set_xlabel('Training Step', fontsize=24, fontweight='bold')
    ax.set_ylabel('Activation Time (ms)', fontsize=24, fontweight='bold')
    ax.set_title('Transformer (BERT) Activation Timing by Function', fontsize=28, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=14, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('transformer_activation_timing.pdf', bbox_inches='tight', facecolor='white')
    
    return fig, ax

def create_transformer_validation_plot(df):
    """Create Transformer validation accuracy plot"""
    if df is None or len(df) == 0:
        return None, None
        
    setup_plotting_style()
    fig, ax = plt.subplots(figsize=(14, 10))
    
    colors = sns.color_palette("husl", len(df['activation'].unique()))
    
    for i, activation in enumerate(sorted(df['activation'].unique())):
        activation_data = df[df['activation'] == activation].copy()
        activation_data = activation_data.sort_values('step')
        
        if 'val_acc' in activation_data.columns:
            ax.plot(activation_data['step'], activation_data['val_acc'] * 100, 
                    label=activation, color=colors[i], linewidth=3, alpha=0.8)
    
    ax.set_xlabel('Training Step', fontsize=24, fontweight='bold')
    ax.set_ylabel('Validation Accuracy (%)', fontsize=24, fontweight='bold')
    ax.set_title('Transformer (BERT) Validation Accuracy by Activation Function', fontsize=28, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=14, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('transformer_validation_accuracy.pdf', bbox_inches='tight', facecolor='white')
    
    return fig, ax

# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def main():
    """Main analysis function for all models"""
    print("🚀 Comprehensive Neural Network Analysis")
    print("=" * 80)
    
    generated_files = []
    
    # MLP Analysis
    print("\n📊 ANALYZING MLP (Multi-Layer Perceptron)")
    print("-" * 50)
    mlp_timing_df, mlp_training_df = load_mlp_data()
    
    if mlp_timing_df is not None:
        create_mlp_timing_plot(mlp_timing_df)
        generated_files.append('mlp_activation_timing.pdf')
        print("✅ MLP timing plot created")
        
    if mlp_training_df is not None:
        create_mlp_validation_plot(mlp_training_df)
        generated_files.append('mlp_validation_accuracy.pdf')
        print("✅ MLP validation accuracy plot created")
    
    # CNN Analysis
    print("\n📊 ANALYZING CNN (ResNet-18)")
    print("-" * 50)
    cnn_timing_df, cnn_training_df = load_cnn_data()
    
    if cnn_timing_df is not None:
        create_cnn_timing_plot(cnn_timing_df)
        generated_files.append('cnn_activation_timing.pdf')
        print("✅ CNN timing plot created")
        
    if cnn_training_df is not None:
        create_cnn_validation_plot(cnn_training_df)
        generated_files.append('cnn_validation_accuracy.pdf')
        print("✅ CNN validation accuracy plot created")
    
    # GNN Analysis
    print("\n📊 ANALYZING GNN (Graph Convolutional Network)")
    print("-" * 50)
    gnn_timing_df, gnn_training_df = load_gnn_data()
    
    if gnn_timing_df is not None:
        create_gnn_timing_plot(gnn_timing_df)
        generated_files.append('gnn_activation_timing.pdf')
        print("✅ GNN timing plot created")
        
    if gnn_training_df is not None:
        create_gnn_validation_plot(gnn_training_df)
        generated_files.append('gnn_validation_accuracy.pdf')
        print("✅ GNN validation accuracy plot created")
    
    # Transformer Analysis
    print("\n📊 ANALYZING TRANSFORMER (BERT)")
    print("-" * 50)
    transformer_timing_df, transformer_training_df = load_transformer_data()
    
    if transformer_timing_df is not None:
        create_transformer_timing_plot(transformer_timing_df)
        generated_files.append('transformer_activation_timing.pdf')
        print("✅ Transformer timing plot created")
        
    if transformer_training_df is not None:
        create_transformer_validation_plot(transformer_training_df)
        generated_files.append('transformer_validation_accuracy.pdf')
        print("✅ Transformer validation accuracy plot created")
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ COMPREHENSIVE ANALYSIS COMPLETE!")
    print("📁 Generated files:")
    for file in generated_files:
        print(f"   - {file}")
    print(f"\n📈 Total PDF files generated: {len(generated_files)}")
    
    return generated_files

if __name__ == "__main__":
    main()
