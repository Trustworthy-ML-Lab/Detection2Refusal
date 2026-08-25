import matplotlib.pyplot as plt
import pandas as pd
import os

model_list = ["guanaco", "qwen2.5", "mistral", "llama2", "qwen3", "llama3"]

# Set up a 2x3 grid of subplots with slightly more vertical space for large text
fig, axes = plt.subplots(3, 2, figsize=(14, 16), sharex=True)
axes = axes.flatten()  # Flatten into a 1D array for easier indexing

# Store line handles for the global legend
lines = []
labels = []

for i, model_name in enumerate(model_list):
    ax = axes[i]
    
    # Construct expected file paths
    targeted_path = f"refusal_contribution_{model_name}.csv"
    random_path = f"refusal_contribution_random_summary_{model_name}.csv"
    
    if os.path.exists(targeted_path) and os.path.exists(random_path):
        # Load datasets
        df_target = pd.read_csv(targeted_path)
        df_rand = pd.read_csv(random_path)
        
        # --- FIX 1: Sort by X-axis to ensure fill_between plots cleanly ---
        df_target = df_target.sort_values('neuron_zero_percent')
        df_rand = df_rand.sort_values('random_zero_percent')
        
        # --- FIX 2: Restrict data up to 5% ---
        df_target = df_target[df_target['neuron_zero_percent'] <= 5]
        df_rand = df_rand[df_rand['random_zero_percent'] <= 5]
        
        # 1. Plot Random Baseline (Thicker line, larger markers)
        line_rand, = ax.plot(df_rand['random_zero_percent'], df_rand['mean_refusal_contribution'], 
                             color='#7f8c8d', linestyle='--', marker='o', markersize=8, linewidth=2.5, alpha=0.8,
                             label='Random Baseline')
        
        # The Standard Deviation Plot
        ax.fill_between(df_rand['random_zero_percent'], 
                        df_rand['mean_refusal_contribution'] - df_rand['std_refusal_contribution'], 
                        df_rand['mean_refusal_contribution'] + df_rand['std_refusal_contribution'], 
                        color='#7f8c8d', alpha=0.15)
        
        # 2. Plot Targeted Removal (Thick line, large square markers)
        line_target, = ax.plot(df_target['neuron_zero_percent'], df_target['avg_refusal_contribution'], 
                               color='#e74c3c', linestyle='-', marker='s', markersize=9, linewidth=3.5,
                               label='Targeted Removal')
        
        # Grab handles from the first valid plot for the global legend
        if not lines:
            lines = [line_target, line_rand]
            labels = [line_target.get_label(), line_rand.get_label()]
        
        # Subplot Title (Significantly larger for paper sections)
        ax.set_title(f"{model_name}".capitalize(), fontsize=35, fontweight='bold', pad=10)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Tick configurations
        ax.set_xticks(df_target['neuron_zero_percent'].unique())
        ax.set_xlim(left=0, right=3) 
        ax.tick_params(axis='both', which='major', labelsize=22)
            
    else:
        ax.text(0.5, 0.5, f"Missing files for\n{model_name}", 
                ha='center', va='center', color='red', fontsize=14, fontweight='bold')
        ax.set_title(f"Model: {model_name} (No Data)", fontsize=16, fontweight='bold')

# --- Global Typography Updates for LaTeX/Overleaf ---
fig.supxlabel('Removal Percentage (%)', fontsize=50, fontweight='bold', y=0.01)
fig.supylabel('Refusal Head Contribution', fontsize=50, fontweight='bold', x=0.01)

# Main Figure Title
plt.suptitle('Targeted vs. Random Neuron Ablation', 
             fontsize=50, fontweight='bold', y=0.98)

# --- NEW: Global Legend placed just below the figure title ---
fig.legend(
    lines, 
    labels, 
    loc='upper center', 
    bbox_to_anchor=(0.5, 0.92),  # Positioned just below the main title (y=0.98)
    ncol=2,                      # Places labels side-by-side
    fontsize=40,                 # High visibility for papers
    frameon=True, 
    facecolor='white', 
    framealpha=0.9
)

# Adjusted rect top (0.86) to leave an empty band at the top for the title and legend
plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.86])
plt.savefig('all_models_circuit_validation.png', dpi=300, bbox_inches='tight')
# plt.show()