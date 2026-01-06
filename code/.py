"""
Denmark DSCR Comparison Plot - All Scenarios and Mechanisms
Generates point plot showing debt service coverage ratio across baseline, moderate, and high scenarios
Output: PDF (vector format, non-pixelated) + PNG preview
"""

import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# DATA INPUT - Modify these values if needed
# ============================================================================

data = {
    'Baseline': {
        'Merchant': 1.675,
        'FiP': 2.943,
        'Flat CfD': 2.662,
        'Tech-Specific CfD': 2.916
    },
    'Moderate': {
        'Merchant': 1.582,
        'FiP': 2.913,
        'Flat CfD': 2.648,
        'Tech-Specific CfD': 2.913
    },
    'High': {
        'Merchant': 1.557,
        'FiP': 2.846,
        'Flat CfD': 2.648,
        'Tech-Specific CfD': 2.940
    }
}

# ============================================================================
# PLOT CONFIGURATION - Modify colors/markers/sizes as needed
# ============================================================================

# Professional, colorblind-friendly color palette
colors = {
    'Merchant': '#D55E00',           # Orange-red
    'FiP': '#0072B2',                # Blue
    'Flat CfD': '#009E73',           # Teal
    'Tech-Specific CfD': '#CC79A7'   # Purple-pink
}

# Different marker shapes for each mechanism
markers = {
    'Merchant': 'o',           # Circle
    'FiP': 's',                # Square
    'Flat CfD': '^',           # Triangle up
    'Tech-Specific CfD': 'D'   # Diamond
}

# ============================================================================
# PLOTTING
# ============================================================================

# Set up figure
plt.figure(figsize=(10, 7))
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

# Scenario positions on x-axis
scenarios = ['Baseline', 'Moderate', 'High']
x_positions = np.arange(len(scenarios))

# Plot each mechanism
for mechanism in ['Merchant', 'FiP', 'Flat CfD', 'Tech-Specific CfD']:
    y_values = [data[scenario][mechanism] for scenario in scenarios]
    
    plt.scatter(x_positions, y_values, 
                s=150,                          # Marker size
                color=colors[mechanism],        # Color
                marker=markers[mechanism],      # Shape
                label=mechanism,                # Legend label
                alpha=0.9,                      # Transparency
                edgecolors='black',             # Black outline
                linewidths=1.5,                 # Outline width
                zorder=3)                       # Layer order

# Add covenant threshold line (DSCR = 1.25)
plt.axhline(y=1.25, color='red', linestyle='--', linewidth=2, 
            label='DSCR Covenant Threshold (1.25)', zorder=2, alpha=0.7)

# Labels and title
plt.xlabel('Electricity Price Scenario', fontsize=13, fontweight='bold')
plt.ylabel('Average DSCR', fontsize=13, fontweight='bold')
plt.title('Debt Service Coverage Ratio: Support Mechanism Comparison (Denmark)', 
          fontsize=14, fontweight='bold', pad=20)

# X-axis configuration
plt.xticks(x_positions, scenarios, fontsize=12)

# Y-axis configuration
plt.ylim(1.0, 3.2)



# Legend
plt.legend(loc='upper right', fontsize=10, framealpha=0.95, 
           edgecolor='black', fancybox=False, shadow=False)

# Tight layout to avoid cutoff
plt.tight_layout()

# ============================================================================
# SAVE OUTPUTS - MODIFY PATHS AS NEEDED
# ============================================================================

# Save as PDF (vector format - perfect for thesis, scalable, non-pixelated)
pdf_path = 'D:/Thesis_Project/thesis_data/Plot/denmark_dscr_comparison.pdf'
plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')

# Save as PNG (high resolution preview)
png_path = 'D:/Thesis_Project/thesis_data/Plot/denmark_dscr_comparison.png'
plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')

print("✓ Plot saved successfully!")
print(f"  PDF: {pdf_path}")
print(f"  PNG: {png_path}")

# Show plot (optional - comment out if you don't want it to display)
plt.show()

# ============================================================================
# VERIFICATION - Print data summary
# ============================================================================

print("\n" + "="*60)
print("DSCR VALUES VERIFICATION")
print("="*60)
for scenario in scenarios:
    print(f"\n{scenario} Scenario:")
    for mechanism in ['Merchant', 'FiP', 'Flat CfD', 'Tech-Specific CfD']:
        dscr = data[scenario][mechanism]
        breach_status = "⚠️ BREACH" if dscr < 1.25 else "✓ OK"
        print(f"  {mechanism:20s}: {dscr:.3f}  {breach_status}")