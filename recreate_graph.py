import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Set much larger font sizes globally to match analyze_all_models.py
plt.rcParams.update({
    'font.size': 22,
    'axes.titlesize': 28,
    'axes.labelsize': 24,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 14,
    'figure.titlesize': 32
})

def setup_plotting_style():
    """Set up consistent plotting style"""
    sns.set_style("whitegrid")
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'

# Set up the plotting style to match analyze_all_models.py
setup_plotting_style()

# Set up the figure with similar styling
plt.figure(figsize=(14, 10))

# Define x-axis range to show the full lines
x = np.linspace(0, 25000, 1000)

# Define the curves based on the exact equations and color mapping
# CV32E40P (red dot) - Calculate slope from the point to ensure line goes through it
# Point: (3509, 0.00586063421), so slope = y/x = 0.00586063421/3509 ≈ 0.000001670172163
cv32e40p_slope = 0.00586063421 / 3509
cv32e40p_y = cv32e40p_slope * x

# Ibex (blue dot) - blue equation: y = 0.0000001148621237x  
ibex_y = 0.0000001148621237 * x

# CGRA (green dot) - green equation: y = 0.000102499550414x
cgra_y = 0.000102499550414 * x

# ASIC (purple dot) - purple equation: y = 0.0000523615038x
asic_y = 0.0000523615038 * x

# Plot the lines without labels (we'll add text annotations instead)
plt.plot(x, cv32e40p_y, 'r-', linewidth=2, alpha=0.8)
plt.plot(x, ibex_y, 'b-', linewidth=2, alpha=0.8)
plt.plot(x, cgra_y, 'g-', linewidth=2, alpha=0.8)
plt.plot(x, asic_y, color='purple', linewidth=2, alpha=0.8)

# Add the exact coordinate points with different shapes
plt.plot(3509, 0.00586063421, 's', color='red', markersize=10, alpha=0.9, markeredgecolor='darkred', markeredgewidth=1)  # CV32E40P point - square
plt.plot(2444, 0.000280723030302, '^', color='blue', markersize=10, alpha=0.9, markeredgecolor='darkblue', markeredgewidth=1)  # Ibex point - triangle
plt.plot(6117.1, 0.627, 'D', color='green', markersize=10, alpha=0.9, markeredgecolor='darkgreen', markeredgewidth=1)  # CGRA point - diamond
plt.plot(19098, 1, 'o', color='purple', markersize=10, alpha=0.9, markeredgecolor='darkmagenta', markeredgewidth=1)  # ASIC point - circle

# Set up the axes with log scaling
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-6, 1e2)  # Smaller y-axis range
plt.xlim(1000, 25000)  # Reduced blank space on left

# Add grid
plt.grid(True, alpha=0.3, which='both')

# Add sloped text annotations along the lines
# Calculate proper rotation angles based on actual slopes
import math

# Calculate the visual angle for each line on the log-log plot
# For log-log plots, we need to account for the log scaling on both axes

def get_log_log_angle(x1, y1, x2, y2):
    """Calculate the visual angle of a line on a log-log plot"""
    # Convert to log space
    log_x1, log_y1 = math.log10(x1), math.log10(y1)
    log_x2, log_y2 = math.log10(x2), math.log10(y2)
    # Calculate angle in log space
    return math.degrees(math.atan2(log_y2 - log_y1, log_x2 - log_x1))

# CV32E40P annotation (red line) - slope = 0.000016701721632
cv32e40p_angle = get_log_log_angle(1000, cv32e40p_slope * 1000, 25000, cv32e40p_slope * 25000) - 38
cv32e40p_x, cv32e40p_y_val = 5000, cv32e40p_slope * 5000
plt.text(cv32e40p_x, cv32e40p_y_val, 'CV32E40P + FPU', fontsize=20, fontweight='bold', 
         color='darkred', rotation=cv32e40p_angle, ha='center', va='bottom')

# Ibex annotation (blue line) - slope = 0.0000001148621237
ibex_angle = get_log_log_angle(1000, 0.0000001148621237 * 1000, 25000, 0.0000001148621237 * 25000) - 38
ibex_x, ibex_y_val = 8000, 0.0000001148621237 * 8000
plt.text(ibex_x, ibex_y_val, 'Ibex + SoftFP', fontsize=20, fontweight='bold', 
         color='darkblue', rotation=ibex_angle, ha='center', va='bottom')

# CGRA annotation (green line) - slope = 0.000102499550414
cgra_angle = get_log_log_angle(1000, 0.000102499550414 * 1000, 25000, 0.000102499550414 * 25000) - 38
cgra_x, cgra_y_val = 4000, 0.000102499550414 * 4000
plt.text(cgra_x, cgra_y_val, 'CGRA + Flopoco', fontsize=20, fontweight='bold', 
         color='darkgreen', rotation=cgra_angle, ha='center', va='bottom')

# ASIC annotation (purple line) - slope = 0.0000523615038
asic_angle = get_log_log_angle(1000, 0.0000523615038 * 1000, 25000, 0.0000523615038 * 25000) - 38
asic_x, asic_y_val = 15000, 0.0000523615038 * 15000  # Moved further right
plt.text(asic_x, asic_y_val, 'ASIC + Flopoco', fontsize=20, fontweight='bold', 
         color='darkmagenta', rotation=asic_angle, ha='center', va='top')  # Changed to va='top'

# Labels and title
plt.xlabel('Area (um^2)', fontsize=24, fontweight='bold')
plt.ylabel('Throughput', fontsize=24, fontweight='bold')
plt.title('Throughput vs Area', fontsize=28, fontweight='bold', pad=20)

# Improve layout
plt.tight_layout()

# Save the plot
plt.savefig('throughput_vs_area.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('throughput_vs_area.pdf', bbox_inches='tight', facecolor='white')

print("📊 Graph recreated and saved as recreated_graph.png and .pdf")
plt.show()
