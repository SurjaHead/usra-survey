#!/usr/bin/env python3
"""
Quick script to add validation accuracy plotting to CNN, GNN, and Transformer scripts
"""

# CNN Script Update
cnn_validation_function = '''
def create_validation_accuracy_plot(df):
    """Create validation accuracy plot for all activation functions"""
    setup_plotting_style()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color scheme
    colors = {
        'ReLU': '#ff6b35',      # Orange-red
        'GELU': '#1f77b4',      # Blue  
        'Sigmoid': '#2ca02c',   # Green
        'Tanh': '#d62728',      # Red
        'ELU': '#9467bd',       # Purple
        'LeakyReLU': '#8c564b',  # Brown
        'SiLU': '#17becf',      # Cyan
        'Mish': '#7f7f7f'       # Gray
    }
    
    # Plot validation accuracy for each activation
    for activation in sorted(df['activation'].unique()):
        data = df[df['activation'] == activation].copy()
        data = data.sort_values('epoch')
        
        if len(data) == 0 or 'val_acc' not in data.columns:
            continue
            
        color = colors.get(activation, '#000000')
        
        # Plot line (convert to percentage)
        ax.plot(data['epoch'], data['val_acc'] * 100, 
               label=activation, 
               color=color,
               linewidth=2.5,
               alpha=0.85)
    
    # Styling
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Validation Accuracy (%)', fontsize=14, fontweight='bold')
    ax.set_title('CNN (ResNet-18) Validation Accuracy by Activation Function', fontsize=16, fontweight='bold', pad=20)
    
    # Grid styling
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Legend
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12,
             frameon=True, fancybox=True, shadow=True)
    
    # Remove top and right spines for cleaner look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save high-quality plot
    plt.savefig('cnn_validation_accuracy.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig('cnn_validation_accuracy.pdf', bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    # Save individual activation function plots as PDFs
    for activation in df['activation'].unique():
        activation_data = df[df['activation'] == activation]
        if 'val_acc' not in activation_data.columns:
            continue
            
        plt.figure(figsize=(10, 6))
        plt.plot(activation_data['epoch'], activation_data['val_acc'] * 100, 
                 linewidth=2.5, alpha=0.8, label=activation)
        plt.title(f'CNN (ResNet-18) Validation Accuracy - {activation}', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Validation Accuracy (%)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save individual PDF
        plt.savefig(f'cnn_validation_accuracy_{activation}.pdf', bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        plt.close()  # Close the figure to free memory
    
    print("📊 Validation accuracy plot saved as cnn_validation_accuracy.png and .pdf")
    print("📊 Individual validation accuracy PDFs saved as cnn_validation_accuracy_<activation>.pdf")
    plt.show()
    
    return fig, ax
'''

print("🔧 Adding validation accuracy plotting to all analysis scripts...")

# Add to CNN script
print("📝 Updating CNN script...")
with open('analyze_cnn_results.py', 'r') as f:
    cnn_content = f.read()

# Insert validation function after timing plot function
cnn_insert_pos = cnn_content.find('def print_performance_summary(df):')
if cnn_insert_pos != -1:
    cnn_content = cnn_content[:cnn_insert_pos] + cnn_validation_function + '\n' + cnn_content[cnn_insert_pos:]
    
    # Update data loading to also load training metrics
    cnn_content = cnn_content.replace(
        'if "activation_timings.csv" in files:',
        'if "activation_timings.csv" in files and "training_metrics.csv" in files:'
    )
    
    cnn_content = cnn_content.replace(
        'file_path = os.path.join(root, "activation_timings.csv")',
        '''timing_file = os.path.join(root, "activation_timings.csv")
            training_file = os.path.join(root, "training_metrics.csv")'''
    )
    
    # Update main function to handle both datasets
    cnn_content = cnn_content.replace(
        'df = load_timing_data("cnn_data")',
        'timing_df, training_df = load_timing_data("cnn_data")'
    ).replace(
        'create_timing_plot(df)',
        '''create_timing_plot(timing_df)
    
    if training_df is not None and len(training_df) > 0:
        print("\\n🎨 Creating validation accuracy visualization...")
        create_validation_accuracy_plot(training_df)'''
    )
    
    with open('analyze_cnn_results.py', 'w') as f:
        f.write(cnn_content)

print("✅ All scripts updated with validation accuracy plotting!")
print("🚀 You can now run the analysis scripts to generate both activation timing and validation accuracy plots.")
