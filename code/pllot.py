import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
# DATA INPUT SECTION - FILL IN YOUR VALUES HERE
# ============================================================================

# Support mechanisms (x-axis labels)
mechanisms = ['Capability CfD', 'Tech-Specific CfD', 'SDE+']

# NPV values for each mechanism (in Millions €)
npv_values = [-1976, -1548, -778]  # Replace with your actual values

# Colors for each bar
colors = ['#3498db', '#e67e22', '#2ecc71']  # Blue, Orange, Green

# ============================================================================
# PLOT SETUP SECTION
# ============================================================================

# Create figure
fig, ax = plt.subplots(figsize=(10, 7))

# ============================================================================
# CREATE BARS SECTION
# ============================================================================

# Create bars
bars = ax.bar(mechanisms, npv_values, color=colors, edgecolor='black', linewidth=1.2)

# ============================================================================
# ADD VALUE LABELS ON TOP OF BARS
# ============================================================================

for i, (bar, value) in enumerate(zip(bars, npv_values)):
    height = bar.get_height()
    # Position label above bar for positive values, below for negative
    if value >= 0:
        va_position = 'bottom'
        y_position = height
    else:
        va_position = 'top'
        y_position = height
    
    ax.text(bar.get_x() + bar.get_width()/2., y_position,
            f'{value}M',
            ha='center', va=va_position, fontsize=11, fontweight='bold')

# ============================================================================
# FORMATTING SECTION
# ============================================================================

# Add labels and title
ax.set_ylabel('NPV (Millions €)', fontsize=13, fontweight='bold')
ax.set_xlabel('Support Mechanism', fontsize=13, fontweight='bold')
ax.set_title('Net Present Value - NL: High Scenario', 
             fontsize=15, fontweight='bold', pad=15)

# Set x-axis labels without rotation for cleaner look
ax.set_xticklabels(mechanisms, fontsize=11)

# Add horizontal line at y=0
ax.axhline(y=0, color='black', linewidth=1)

# Adjust layout first
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)  # Increased bottom margin to prevent overlap

# Add subtitle below the plot with proper spacing
fig.text(0.5, 0.04, '(a) High price scenario', 
         ha='center', fontsize=12, style='italic')

# ============================================================================
# SAVE AND DISPLAY SECTION
# ============================================================================

# Save as PDF (vector format - no pixels!)
plt.savefig('D:/Thesis_Project/thesis_data/Plot/npv_high_nl.pdf', 
            format='pdf', bbox_inches='tight')
plt.show()