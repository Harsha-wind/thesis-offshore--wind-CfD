import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# MECHANISM COMPARISON: SUPPORT EFFECTIVENESS
# High Scenario - Denmark
# =============================================================================

print("="*80)
print("MECHANISM COMPARISON VISUALIZATION")
print("="*80)

# =============================================================================
# INPUT DATA - Replace with your actual values
# =============================================================================

# Data structure: {Metric: {Mechanism: Value}}
data = {
    'CV Reduction (%)': {
        'FiP': 52.8,  # Replace with your FiP value
        'Cap-CfD\n(Flat Avg)': 72.8,  # Replace with your flat average value
        'Cap-CfD\n(Tech-Spec)': 76.5
    },
    'Downside CoV\nReduction (%)': {
        'FiP': 89.3,  # Replace
        'Cap-CfD\n(Flat Avg)': 83.0,  # Replace
        'Cap-CfD\n(Tech-Spec)': 88.5
    },
    'VaR Improvement (%)': {
        'FiP': 412.5,  # Replace
        'Cap-CfD\n(Flat Avg)': 381.5,  # Replace
        'Cap-CfD\n(Tech-Spec)': 472.6
    },
    'CVaR Improvement (%)': {
        'FiP': 815.9,  # Replace
        'Cap-CfD\n(Flat Avg)': 736.8,  # Replace
        'Cap-CfD\n(Tech-Spec)': 906.8
    }
}

# Convert to DataFrame
df_compare = pd.DataFrame(data).T

# =============================================================================
# VISUALIZATION 1: Grouped Bar Chart
# =============================================================================

fig, ax = plt.subplots(figsize=(14, 8))

x = np.arange(len(df_compare.index))
width = 0.25

colors = ['#3498db', '#e67e22', '#2ecc71']  # Blue, Orange, Green

bars1 = ax.bar(x - width, df_compare['FiP'], width, 
               label='FiP', color=colors[0], alpha=0.8, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x, df_compare['Cap-CfD\n(Flat Avg)'], width, 
               label='Capability-CfD (Flat Average)', color=colors[1], alpha=0.8, edgecolor='black', linewidth=1.5)
bars3 = ax.bar(x + width, df_compare['Cap-CfD\n(Tech-Spec)'], width, 
               label='Capability-CfD (Tech-Specific)', color=colors[2], alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)

ax.set_ylabel('Improvement (%)', fontsize=13, fontweight='bold')
ax.set_title('Support Mechanism Effectiveness: Key Performance Metrics', 
             fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(df_compare.index, fontsize=11, fontweight='bold')
ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim(0, max(df_compare.max()) * 1.15)

plt.tight_layout()
plt.savefig('D:/Thesis_Project/thesis_data/results/DK/mechanism_comparison_improvements.png', 
            dpi=300, bbox_inches='tight')
plt.show()

print("✓ Visualization 1 saved: mechanism_comparison_improvements.png")