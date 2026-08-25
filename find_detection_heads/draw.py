import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================
# Load data
# =========================
df = pd.read_csv("drawing_result.csv")
df = df[df["model"] != "vicuna"]
df = df[df["model"] != "llama3.1"]

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
plt.title(r"Safety Rate vs Intervention Factor $\alpha_{det}$", fontsize=28)
plt.xlabel(r"Detection Heads Intervention Factor $\alpha_{det}$", fontsize=25)
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
plt.savefig("Detection_Intervention.png")