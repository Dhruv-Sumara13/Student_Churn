"""
Create visual comparison of old vs new semester weighting
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parents[1] / 'reports' / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Setup
semesters = [1, 2, 3, 4, 5, 6]
old_weights = [0.10, 0.10, 0.10, 0.00, 0.00, 0.00]
new_weights = [0.20, 0.15, 0.05, 0.00, 0.00, 0.00]
historical_churn = [100, 80.9, 3.2, 0, 0, 0]

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Weight Comparison
x = np.arange(len(semesters))
width = 0.35

bars1 = ax1.bar(x - width/2, old_weights, width, label='Old (Fixed 10%)', 
                color='#9B6A1A', alpha=0.7)
bars2 = ax1.bar(x + width/2, new_weights, width, label='New (Adaptive)', 
                color='#C8A96E', alpha=0.9)

ax1.set_xlabel('Semester', fontsize=12, fontweight='bold')
ax1.set_ylabel('Semester Signal Weight', fontsize=12, fontweight='bold')
ax1.set_title('Semester Weighting: Old vs New', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels([f'Sem {s}' for s in semesters])
ax1.legend(fontsize=11)
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim(0, 0.25)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.0%}', ha='center', va='bottom', fontsize=9)

# Plot 2: Historical Churn Rate
ax2.plot(semesters, historical_churn, marker='o', linewidth=3, 
         markersize=10, color='#8B2525', label='Historical Churn Rate')
ax2.fill_between(semesters, historical_churn, alpha=0.2, color='#8B2525')

ax2.set_xlabel('Semester', fontsize=12, fontweight='bold')
ax2.set_ylabel('Historical Churn Rate (%)', fontsize=12, fontweight='bold')
ax2.set_title('Why Adaptive Weighting Makes Sense', fontsize=14, fontweight='bold')
ax2.set_xticks(semesters)
ax2.set_xticklabels([f'Sem {s}' for s in semesters])
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-5, 110)

# Add annotations
ax2.annotate('CRITICAL\n100% churn', xy=(1, 100), xytext=(1.3, 85),
            arrowprops=dict(arrowstyle='->', color='#8B2525', lw=2),
            fontsize=10, fontweight='bold', color='#8B2525')
ax2.annotate('HIGH RISK\n80.9% churn', xy=(2, 80.9), xytext=(2.3, 65),
            arrowprops=dict(arrowstyle='->', color='#9B6A1A', lw=2),
            fontsize=10, fontweight='bold', color='#9B6A1A')
ax2.annotate('LOW RISK\n3.2% churn', xy=(3, 3.2), xytext=(3.5, 20),
            arrowprops=dict(arrowstyle='->', color='#2D6A4F', lw=2),
            fontsize=10, fontweight='bold', color='#2D6A4F')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'semester_weighting_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Chart saved as 'semester_weighting_comparison.png'")

# Create second chart: Prediction impact
fig2, ax3 = plt.subplots(figsize=(10, 6))

base_risks = [0.20, 0.30, 0.40, 0.50]
colors = ['#2D6A4F', '#9B6A1A', '#8B2525', '#5C0A0A']
labels = ['20% base', '30% base', '40% base', '50% base']

for base_risk, color, label in zip(base_risks, colors, labels):
    old_finals = []
    new_finals = []
    
    for sem in semesters:
        # Old model
        if sem <= 3:
            hist_rate = {1: 1.0, 2: 0.809, 3: 0.032}[sem]
            old_final = 0.90 * base_risk + 0.10 * hist_rate
        else:
            old_final = base_risk
        old_finals.append(old_final * 100)
        
        # New model
        if sem <= 3:
            weights = {1: 0.20, 2: 0.15, 3: 0.05}
            hist_rate = {1: 1.0, 2: 0.809, 3: 0.032}[sem]
            new_final = (1 - weights[sem]) * base_risk + weights[sem] * hist_rate
        else:
            new_final = base_risk
        new_finals.append(new_final * 100)
    
    ax3.plot(semesters, old_finals, '--', alpha=0.5, color=color, linewidth=2)
    ax3.plot(semesters, new_finals, '-', marker='o', color=color, linewidth=2.5, 
             markersize=8, label=label)

ax3.set_xlabel('Semester', fontsize=12, fontweight='bold')
ax3.set_ylabel('Final Churn Probability (%)', fontsize=12, fontweight='bold')
ax3.set_title('Prediction Impact: Old (dashed) vs New (solid) Model', 
              fontsize=14, fontweight='bold')
ax3.set_xticks(semesters)
ax3.set_xticklabels([f'Sem {s}' for s in semesters])
ax3.legend(title='Base Admission Risk', fontsize=10, title_fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.axhline(y=45, color='#C8A96E', linestyle=':', linewidth=2, label='Threshold (45%)')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'prediction_impact_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Chart saved as 'prediction_impact_comparison.png'")

print("\n" + "="*60)
print("VISUAL COMPARISON CHARTS CREATED")
print("="*60)
print("\n1. semester_weighting_comparison.png")
print("   - Shows old vs new weight distribution")
print("   - Shows historical churn rate justification")
print("\n2. prediction_impact_comparison.png")
print("   - Shows how predictions change for different base risks")
print("   - Dashed lines = old model, Solid lines = new model")
print("\nKey observation: New model is more aggressive in Sem 1-2")
print("and more conservative in Sem 3, matching actual risk patterns.")
