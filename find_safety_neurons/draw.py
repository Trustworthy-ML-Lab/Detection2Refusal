# import pandas as pd
# import matplotlib.pyplot as plt

# # Load CSV
# df = pd.read_csv("safety_neuron_intervention_result.csv")
# df = df[df["factor"] >= 1.0]
# df = df[df['model'] == 'llama3.1']

# # Loop over each percent value
# for pct in sorted(df["percent"].unique()):
#     plt.figure(figsize=(10, 6))

#     # Filter only rows with this percent
#     sub = df[df["percent"] == pct]

#     # Plot a line for each model
#     for model, mdf in sub.groupby("model"):
#         plt.plot(
#             mdf["factor"],
#             mdf["safe_guard"],
#             marker="o",
#             linewidth=2,
#             label=model
#         )

#     plt.xlabel("Factor", fontsize=12)
#     plt.ylabel("Safe Guard Score", fontsize=12)
#     plt.title(f"Safe Guard vs Factor (percent = {pct})", fontsize=14)
#     plt.legend()
#     plt.tight_layout()

#     # Save figure
#     output_name = f"intervention_plotting/safe_guard_percent_{pct}.png"
#     plt.savefig(output_name, dpi=300)
#     plt.close()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================
# Load data
# =========================
df = pd.read_csv("drawing_result.csv")
df = df[df["model"] != "vicuna"]
df = df[df["model"] != "llama3.1"]
df = df[df["factor"] <= 2.0]

# =========================
# Fixed color map per model
# =========================
color_map = {
    "llama2": "#1f77b4",
    "llama3": "#ff7f0e",
    "llama3.1": "#2ca02c",
    "mistral": "#d62728",
    "guanaco": "#9467bd",
    "qwen2.5": "#8c564b",
    "qwen3": "#e377c2",
}

# =========================
# Plot
# =========================
plt.figure(figsize=(10, 6))

for model, subset in df.groupby("model"):
    subset = subset.sort_values("factor")
    plt.plot(
        subset["factor"],
        subset["safe_guard"],
        marker="o",
        linewidth=2,
        label=model,
        color=color_map.get(model, None),
    )

# =========================
# Labels & styling
# =========================
plt.title(r"Safety Rate vs Intervention Factor $\alpha_{saf}$", fontsize=28)
plt.xlabel(r"Safety Neurons Intervention Factor $\alpha_{saf}$", fontsize=25)
plt.ylabel("Safety Rate", fontsize=25)

xmin = df["factor"].min()
xmax = df["factor"].max()
plt.xticks(np.arange(xmin, xmax + 0.5, 0.5), fontsize=20)
plt.yticks(fontsize=20)


# Legend outside the plot
plt.legend(
    title="Model",
    bbox_to_anchor=(1.02, 0.5),
    loc="center left",
    frameon=False,
    fontsize=23,
    title_fontsize=23
)

plt.grid(False)

plt.tight_layout()
plt.savefig("Safety_Intervention.png")