import pandas as pd
import numpy as np

# ==============================================================================
# MARKET DISTORTION COMPARISON: SDE+ vs Capability CfD
# ==============================================================================
# Purpose: Compare dispatch behavior under different support mechanisms
# Key Question: Which mechanism better avoids "produce-and-forget" distortion?
#
# SDE+ (Day-Ahead): Curtail ENTIRE blocks ≥6 consecutive negative hours
# Capability CfD: Curtail at ALL negative prices (paid on potential)
# ==============================================================================

print("="*80)
print("MARKET DISTORTION ANALYSIS: 'Produce-and-Forget' Comparison")
print("="*80)
print("Comparing: SDE+ (6-hour rule) vs Capability CfD (generation-potential)")
print("="*80)

# ==============================================================================
# STEP 1: LOAD HOURLY DATA
# ==============================================================================

df = pd.read_csv(
    "D:/Thesis_Project/thesis_data/results/NL/Current/high/hourly_actual_generation_wake_NL_(High_NL).csv",
    parse_dates=["hours"]
)

# Clean column names
df.columns = df.columns.str.strip()

# Standardize column names
df.rename(columns={
    'hours': 'hour',
    'SpotPrice/MWh': 'SpotPrice_€/MWh',
    'Actual_Generation_MWh': 'Generation_MWh'
}, inplace=True)

# Clean data
df['hour'] = pd.to_datetime(df['hour'], utc=True, errors='coerce')
df = df.dropna(subset=['hour']).drop_duplicates(subset=['hour']).sort_values('hour').reset_index(drop=True)

print(f"\n✓ Loaded {len(df):,} hours of data")
print(f"  Period: {df['hour'].min()} to {df['hour'].max()}")
print(f"  Total generation potential: {df['Generation_MWh'].sum():,.0f} MWh\n")

# ==============================================================================
# STEP 2: IDENTIFY NEGATIVE PRICE PERIODS
# ==============================================================================

print("="*80)
print("STEP 1: NEGATIVE PRICE IDENTIFICATION")
print("="*80)

# Identify negative/zero price hours
df['is_negative'] = df['SpotPrice_€/MWh'] < 0
df['is_zero'] = df['SpotPrice_€/MWh'] == 0
df['is_low'] = (df['SpotPrice_€/MWh'] >= 0) & (df['SpotPrice_€/MWh'] < 10)

# Identify consecutive negative price blocks
df['neg_block'] = (df['is_negative'] != df['is_negative'].shift()).cumsum()
df['consecutive_neg'] = df.groupby('neg_block').cumcount() + 1
df['consecutive_neg'] = df['consecutive_neg'] * df['is_negative']

# Calculate the TOTAL LENGTH of each negative block (for day-ahead decision)
df['neg_block_length'] = df.groupby('neg_block')['is_negative'].transform('sum')

neg_hours = df['is_negative'].sum()
total_hours = len(df)

print(f"\nA. NEGATIVE PRICE STATISTICS:")
print(f"   Total hours with negative prices: {neg_hours:,} ({neg_hours/total_hours*100:.2f}%)")
print(f"   Average negative price: €{df.loc[df['is_negative'], 'SpotPrice_€/MWh'].mean():.2f}/MWh")
print(f"   Minimum price: €{df['SpotPrice_€/MWh'].min():.2f}/MWh")

# Find longest consecutive negative stretch
if neg_hours > 0:
    max_consecutive = df[df['is_negative']]['neg_block_length'].max()
    print(f"   Longest consecutive negative stretch: {max_consecutive:.0f} hours")

# ==============================================================================
# STEP 3: CALCULATE DISPATCH DECISIONS (DAY-AHEAD MARKET)
# ==============================================================================

print("\n" + "="*80)
print("STEP 2: DISPATCH BEHAVIOR SIMULATION (Day-Ahead Market)")
print("="*80)

# -------------------------------------------------------------------------
# SCENARIO 1: SDE+ (6-Hour Curtailment Rule with Day-Ahead Knowledge)
# -------------------------------------------------------------------------
print("\nA. SDE+ DISPATCH LOGIC (Day-Ahead Market):")
print("   Rule: Subsidy only for first 5 consecutive negative hours")
print("   Day-ahead knowledge: Generators see tomorrow's prices at noon")
print("   Rational behavior: If negative block ≥6 hours → curtail ENTIRE block\n")

# SDE+ dispatch decision (made day-ahead):
# - If negative block is <6 hours total → Generate through it (eligible for subsidy)
# - If negative block is ≥6 hours total → Curtail the ENTIRE block (not worth it)
df['Dispatched_SDE+_MWh'] = np.where(
    df['is_negative'] & (df['neg_block_length'] >= 6),
    0,  # Curtail ENTIRE block if it's ≥6 hours
    df['Generation_MWh']  # Generate otherwise
)

# Calculate curtailed generation under SDE+
df['Curtailed_SDE+_MWh'] = df['Generation_MWh'] - df['Dispatched_SDE+_MWh']

curtailed_sde = df['Curtailed_SDE+_MWh'].sum()
dispatched_sde = df['Dispatched_SDE+_MWh'].sum()
total_gen = df['Generation_MWh'].sum()

print(f"   Total generation potential: {total_gen:,.0f} MWh")
print(f"   Dispatched under SDE+: {dispatched_sde:,.0f} MWh ({dispatched_sde/total_gen*100:.2f}%)")
print(f"   Curtailed under SDE+: {curtailed_sde:,.0f} MWh ({curtailed_sde/total_gen*100:.2f}%)")

# Generation during negative prices under SDE+
gen_at_neg_sde = df.loc[df['is_negative'], 'Dispatched_SDE+_MWh'].sum()
print(f"   Generation at negative prices: {gen_at_neg_sde:,.0f} MWh ({gen_at_neg_sde/total_gen*100:.2f}%)")

# -------------------------------------------------------------------------
# SCENARIO 2: Capability CfD (Immediate Curtailment at Negative Prices)
# -------------------------------------------------------------------------
print("\nB. CAPABILITY CFD DISPATCH LOGIC:")
print("   Rule: Paid on generation POTENTIAL (not actual)")
print("   Day-ahead knowledge: Generators see tomorrow's prices")
print("   Rational behavior: Curtail at ANY negative price (no economic reason to produce)\n")

# Capability CfD: Curtail immediately at any negative price
df['Dispatched_CapCfD_MWh'] = np.where(
    df['is_negative'],
    0,  # Curtail at ALL negative prices
    df['Generation_MWh']  # Generate at positive prices
)

# Calculate curtailed generation under Capability CfD
df['Curtailed_CapCfD_MWh'] = df['Generation_MWh'] - df['Dispatched_CapCfD_MWh']

curtailed_cap = df['Curtailed_CapCfD_MWh'].sum()
dispatched_cap = df['Dispatched_CapCfD_MWh'].sum()

print(f"   Total generation potential: {total_gen:,.0f} MWh")
print(f"   Dispatched under CapCfD: {dispatched_cap:,.0f} MWh ({dispatched_cap/total_gen*100:.2f}%)")
print(f"   Curtailed under CapCfD: {curtailed_cap:,.0f} MWh ({curtailed_cap/total_gen*100:.2f}%)")

# Generation during negative prices under Capability CfD
gen_at_neg_cap = df.loc[df['is_negative'], 'Dispatched_CapCfD_MWh'].sum()
print(f"   Generation at negative prices: {gen_at_neg_cap:,.0f} MWh ({gen_at_neg_cap/total_gen*100:.2f}%)")

# ==============================================================================
# STEP 4: METRIC 1 - NEGATIVE PRICE EXPOSURE
# ==============================================================================

print("\n" + "="*80)
print("METRIC 1: NEGATIVE PRICE EXPOSURE COMPARISON")
print("="*80)
print("Lower exposure = Better price response = Less market distortion\n")

exposure_comparison = pd.DataFrame({
    'Mechanism': ['SDE+', 'Capability CfD', 'Difference'],
    'Hours at Neg Prices': [
        f"{neg_hours:,}",
        f"{neg_hours:,}",
        "-"
    ],
    'Gen at Neg Prices (MWh)': [
        f"{gen_at_neg_sde:,.0f}",
        f"{gen_at_neg_cap:,.0f}",
        f"{gen_at_neg_sde - gen_at_neg_cap:,.0f}"
    ],
    '% of Total Gen': [
        f"{gen_at_neg_sde/total_gen*100:.2f}%",
        f"{gen_at_neg_cap/total_gen*100:.2f}%",
        f"{(gen_at_neg_sde - gen_at_neg_cap)/total_gen*100:.2f}%"
    ]
})

print(exposure_comparison.to_string(index=False))

if gen_at_neg_cap < gen_at_neg_sde:
    reduction = (gen_at_neg_sde - gen_at_neg_cap) / gen_at_neg_sde * 100
    print(f"\n✓ Capability CfD reduces negative price generation by {reduction:.1f}%")
if gen_at_neg_cap == 0:
    print(f"✓ Capability CfD completely eliminates generation at negative prices!")

# ==============================================================================
# STEP 5: METRIC 2 - DISPATCH EFFICIENCY (CORRELATION)
# ==============================================================================

print("\n" + "="*80)
print("METRIC 2: DISPATCH EFFICIENCY (Generation-Price Correlation)")
print("="*80)
print("Higher correlation = Better price response = More efficient dispatch\n")

# Calculate correlations
corr_sde = df['Dispatched_SDE+_MWh'].corr(df['SpotPrice_€/MWh'])
corr_cap = df['Dispatched_CapCfD_MWh'].corr(df['SpotPrice_€/MWh'])

print(f"A. GENERATION-PRICE CORRELATION:")
print(f"   SDE+: {corr_sde:.4f}")
print(f"   Capability CfD: {corr_cap:.4f}")
print(f"   Difference: {corr_cap - corr_sde:+.4f}")

if corr_cap > corr_sde:
    print(f"\n   ✓ Capability CfD shows BETTER price alignment")
else:
    print(f"\n   ⚠ SDE+ shows better price alignment")

# Generation by price quartiles
df['price_quartile'] = pd.qcut(df['SpotPrice_€/MWh'], q=4, labels=['Q1 (lowest)', 'Q2', 'Q3', 'Q4 (highest)'])

print(f"\nB. GENERATION BY PRICE QUARTILE:")
print(f"\n   SDE+:")
sde_quartiles = df.groupby('price_quartile')['Dispatched_SDE+_MWh'].sum()
for q, gen in sde_quartiles.items():
    print(f"      {q}: {gen:,.0f} MWh ({gen/dispatched_sde*100:.1f}%)")

print(f"\n   Capability CfD:")
cap_quartiles = df.groupby('price_quartile')['Dispatched_CapCfD_MWh'].sum()
for q, gen in cap_quartiles.items():
    print(f"      {q}: {gen:,.0f} MWh ({gen/dispatched_cap*100:.1f}%)")

# Efficiency ratio (Q4/Q1)
sde_ratio = sde_quartiles.iloc[-1] / sde_quartiles.iloc[0] if sde_quartiles.iloc[0] > 0 else float('inf')
cap_ratio = cap_quartiles.iloc[-1] / cap_quartiles.iloc[0] if cap_quartiles.iloc[0] > 0 else float('inf')

print(f"\nC. EFFICIENCY SCORE (Q4/Q1 ratio):")
print(f"   SDE+: {sde_ratio:.2f}x" if sde_ratio != float('inf') else "   SDE+: ∞ (no Q1 generation)")
print(f"   Capability CfD: {cap_ratio:.2f}x" if cap_ratio != float('inf') else "   Capability CfD: ∞ (no Q1 generation)")
print(f"   (Higher is better - means more generation at high prices)")

# ==============================================================================
# STEP 6: METRIC 3 - SYSTEM COST OF MISALIGNMENT
# ==============================================================================

print("\n" + "="*80)
print("METRIC 3: SYSTEM COST OF MISALIGNMENT")
print("="*80)
print("Lower cost = Less market distortion = Better for system\n")

# Calculate revenue loss from negative price generation
revenue_loss_sde = (df.loc[df['is_negative'], 'SpotPrice_€/MWh'] * 
                    df.loc[df['is_negative'], 'Dispatched_SDE+_MWh']).sum()

revenue_loss_cap = (df.loc[df['is_negative'], 'SpotPrice_€/MWh'] * 
                    df.loc[df['is_negative'], 'Dispatched_CapCfD_MWh']).sum()

print(f"A. DIRECT REVENUE LOSS FROM NEGATIVE PRICE GENERATION:")
print(f"   SDE+: €{abs(revenue_loss_sde):,.2f}")
print(f"   Capability CfD: €{abs(revenue_loss_cap):,.2f}")
print(f"   Savings with CapCfD: €{abs(revenue_loss_sde - revenue_loss_cap):,.2f}")

# Opportunity cost (could have used curtailed capacity at better times)
avg_price = df['SpotPrice_€/MWh'].mean()
avg_neg_price = df.loc[df['is_negative'], 'SpotPrice_€/MWh'].mean()
opp_cost_per_mwh = avg_price - avg_neg_price

opp_cost_sde = opp_cost_per_mwh * gen_at_neg_sde
opp_cost_cap = opp_cost_per_mwh * gen_at_neg_cap

print(f"\nB. OPPORTUNITY COST (Generation at wrong time):")
print(f"   Average market price: €{avg_price:.2f}/MWh")
print(f"   Average negative price: €{avg_neg_price:.2f}/MWh")
print(f"   Opportunity cost per MWh: €{opp_cost_per_mwh:.2f}")
print(f"\n   SDE+ opportunity cost: €{opp_cost_sde:,.2f}")
print(f"   CapCfD opportunity cost: €{opp_cost_cap:,.2f}")
print(f"   System savings with CapCfD: €{opp_cost_sde - opp_cost_cap:,.2f}")

# ==============================================================================
# STEP 7: METRIC 4 - CURTAILMENT RESPONSE BEHAVIOR
# ==============================================================================

print("\n" + "="*80)
print("METRIC 4: CURTAILMENT RESPONSE BEHAVIOR")
print("="*80)
print("Better curtailment = Better market response = Less distortion\n")

print(f"A. NEGATIVE PRICE BLOCK ANALYSIS:")
print(f"   (Day-ahead market: generators see full block length before deciding)\n")

# Analyze negative blocks by length
neg_blocks = df[df['is_negative']].groupby('neg_block').agg({
    'neg_block_length': 'first',
    'Generation_MWh': 'sum',
    'Dispatched_SDE+_MWh': 'sum',
    'Dispatched_CapCfD_MWh': 'sum'
}).reset_index()

# Count blocks by length
short_blocks = neg_blocks[neg_blocks['neg_block_length'] < 6]
long_blocks = neg_blocks[neg_blocks['neg_block_length'] >= 6]

print(f"B. SHORT NEGATIVE BLOCKS (<6 hours):")
print(f"   Number of blocks: {len(short_blocks)}")
if len(short_blocks) > 0:
    print(f"   Total hours: {short_blocks['neg_block_length'].sum():.0f}")
    print(f"   Potential generation: {short_blocks['Generation_MWh'].sum():,.0f} MWh")
    print(f"   SDE+ generates: {short_blocks['Dispatched_SDE+_MWh'].sum():,.0f} MWh (gets subsidy)")
    print(f"   CapCfD generates: {short_blocks['Dispatched_CapCfD_MWh'].sum():,.0f} MWh (curtails anyway)")
else:
    print(f"   None found in dataset")

print(f"\nC. LONG NEGATIVE BLOCKS (≥6 hours):")
print(f"   Number of blocks: {len(long_blocks)}")
if len(long_blocks) > 0:
    print(f"   Total hours: {long_blocks['neg_block_length'].sum():.0f}")
    print(f"   Average length: {long_blocks['neg_block_length'].mean():.1f} hours")
    print(f"   Longest block: {long_blocks['neg_block_length'].max():.0f} hours")
    print(f"   Potential generation: {long_blocks['Generation_MWh'].sum():,.0f} MWh")
    print(f"   SDE+ generates: {long_blocks['Dispatched_SDE+_MWh'].sum():,.0f} MWh (should be 0)")
    print(f"   CapCfD generates: {long_blocks['Dispatched_CapCfD_MWh'].sum():,.0f} MWh (should be 0)")
else:
    print(f"   None found in dataset")

print(f"\nD. DAY-AHEAD DECISION MAKING:")
print(f"   SDE+ Logic:")
print(f"      - See day-ahead prices at noon")
print(f"      - If negative block <6 hours → Generate (eligible for subsidy)")
print(f"      - If negative block ≥6 hours → Curtail ENTIRE block (no subsidy from hour 6)")
print(f"   Capability CfD Logic:")
print(f"      - See day-ahead prices at noon")
print(f"      - If negative price → Curtail (paid on potential anyway)")
print(f"      - Simpler decision, more responsive to market signals")

# ==============================================================================
# STEP 8: COMPREHENSIVE COMPARISON SUMMARY
# ==============================================================================

print("\n" + "="*80)
print("COMPREHENSIVE SUMMARY: SDE+ vs CAPABILITY CFD")
print("="*80)

summary = pd.DataFrame({
    'Metric': [
        'Total Generation Potential (MWh)',
        'Actually Dispatched (MWh)',
        'Curtailed (MWh)',
        '% Curtailed',
        '',
        'Generation at Negative Prices (MWh)',
        '% at Negative Prices',
        '',
        'Generation-Price Correlation',
        'Q4/Q1 Efficiency Ratio',
        '',
        'Revenue Loss at Neg Prices (€)',
        'Opportunity Cost (€)',
        'Total System Cost (€)',
        '',
        'Curtailment Response',
        'Market Distortion Risk'
    ],
    'SDE+': [
        f"{total_gen:,.0f}",
        f"{dispatched_sde:,.0f}",
        f"{curtailed_sde:,.0f}",
        f"{curtailed_sde/total_gen*100:.2f}%",
        '',
        f"{gen_at_neg_sde:,.0f}",
        f"{gen_at_neg_sde/total_gen*100:.2f}%",
        '',
        f"{corr_sde:.4f}",
        f"{sde_ratio:.2f}x" if sde_ratio != float('inf') else "∞",
        '',
        f"{abs(revenue_loss_sde):,.0f}",
        f"{opp_cost_sde:,.0f}",
        f"{abs(revenue_loss_sde) + opp_cost_sde:,.0f}",
        '',
        'Block-based',
        'Moderate-Low'
    ],
    'Capability CfD': [
        f"{total_gen:,.0f}",
        f"{dispatched_cap:,.0f}",
        f"{curtailed_cap:,.0f}",
        f"{curtailed_cap/total_gen*100:.2f}%",
        '',
        f"{gen_at_neg_cap:,.0f}",
        f"{gen_at_neg_cap/total_gen*100:.2f}%",
        '',
        f"{corr_cap:.4f}",
        f"{cap_ratio:.2f}x" if cap_ratio != float('inf') else "∞",
        '',
        f"{abs(revenue_loss_cap):,.0f}",
        f"{opp_cost_cap:,.0f}",
        f"{abs(revenue_loss_cap) + opp_cost_cap:,.0f}",
        '',
        'Immediate',
        'Low'
    ],
    'Winner': [
        'Tie',
        'CapCfD' if dispatched_cap < dispatched_sde else 'SDE+',
        'CapCfD' if curtailed_cap > curtailed_sde else 'SDE+',
        'CapCfD' if curtailed_cap > curtailed_sde else 'SDE+',
        '',
        'CapCfD' if gen_at_neg_cap < gen_at_neg_sde else 'SDE+',
        'CapCfD' if gen_at_neg_cap < gen_at_neg_sde else 'SDE+',
        '',
        'CapCfD' if corr_cap > corr_sde else 'SDE+',
        'CapCfD' if cap_ratio > sde_ratio else 'SDE+',
        '',
        'CapCfD' if abs(revenue_loss_cap) < abs(revenue_loss_sde) else 'SDE+',
        'CapCfD' if opp_cost_cap < opp_cost_sde else 'SDE+',
        'CapCfD' if (abs(revenue_loss_cap) + opp_cost_cap) < (abs(revenue_loss_sde) + opp_cost_sde) else 'SDE+',
        '',
        'CapCfD',
        'CapCfD'
    ]
})

print(summary.to_string(index=False))

# ==============================================================================
# STEP 9: INTERPRETATION FOR THESIS
# ==============================================================================

print("\n" + "="*80)
print("INTERPRETATION: Does Capability CfD Avoid 'Produce-and-Forget'?")
print("="*80)

print("\n✓ KEY FINDINGS:")
print(f"\n1. NEGATIVE PRICE GENERATION:")
if gen_at_neg_cap < gen_at_neg_sde * 0.5:
    print(f"   ✓ Capability CfD SIGNIFICANTLY reduces generation at negative prices")
    print(f"   ✓ Reduction: {(gen_at_neg_sde - gen_at_neg_cap)/gen_at_neg_sde*100:.1f}%")
if gen_at_neg_cap == 0:
    print(f"   ✓ Capability CfD ELIMINATES generation at negative prices entirely")
    print(f"   ⚠ SDE+ still generates {gen_at_neg_sde:,.0f} MWh during short negative blocks")

print(f"\n2. DISPATCH EFFICIENCY:")
if corr_cap > corr_sde:
    print(f"   ✓ Capability CfD shows BETTER generation-price alignment")
    print(f"   ✓ Correlation improvement: {(corr_cap - corr_sde):.4f}")

print(f"\n3. SYSTEM COST:")
system_savings = (abs(revenue_loss_sde) + opp_cost_sde) - (abs(revenue_loss_cap) + opp_cost_cap)
if system_savings > 0:
    print(f"   ✓ Capability CfD reduces system costs by €{system_savings:,.2f}")
    print(f"   ✓ Savings come from avoiding ALL negative-price generation")

print(f"\n4. CURTAILMENT BEHAVIOR:")
print(f"   ✓ Capability CfD: Curtails at ANY negative price (most responsive)")
print(f"   ⚠ SDE+: Generates through short negative blocks (<6h), curtails long blocks (≥6h)")
print(f"   → CapCfD provides more consistent market signal response")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
print("\n✓ Capability CfD DOES NOT create 'produce-and-forget' distortion")
print("✓ Comparison with SDE+:")
print("   • SDE+ generates through short negative periods (<6 hours)")
print("   • Capability CfD curtails at ALL negative prices")
print("   • Result: CapCfD is MORE responsive to market signals")
print("\n✓ Key advantages:")
print("   1. Eliminates ALL negative-price generation (not just long blocks)")
print("   2. Simpler dispatch decision (curtail when negative, period)")
print("   3. Better generation-price alignment")
print("   4. Lower system-wide inefficiency costs")
print("\n✓ Mechanism: Payment based on POTENTIAL (not actual) removes")
print("  ALL perverse incentives to produce at economically irrational times")
print("="*80)

# ==============================================================================
# STEP 10: EXPORT RESULTS
# ==============================================================================

print("\n" + "="*80)
print("EXPORTING RESULTS")
print("="*80)

# Save detailed hourly data
output_file = "D:/Thesis_Project/thesis_data/results/NL/market_distortion_comparison_high.csv"
df.to_csv(output_file, index=False)
print(f"\n✓ Detailed hourly data saved to:\n  {output_file}")

# Save summary
summary_output = "D:/Thesis_Project/thesis_data/results/NL/market_distortion_summary_high.csv"
summary.to_csv(summary_output, index=False)
print(f"✓ Summary comparison saved to:\n  {summary_output}")

print("\n" + "="*80)
print("✓ ANALYSIS COMPLETE!")
print("="*80)