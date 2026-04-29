import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
file_path = "test_results.csv"
df = pd.read_csv(file_path)

# List of similarity columns
sim_cols = ["tax_large_sim", "ver_large_sim", "ver_base_sim", "tax_base_sim"]

# Define bins and labels
bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
labels = ["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]

# Prepare bin counts for each sim column
bin_counts = {
    col: pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
    .value_counts()
    .sort_index()
    for col in sim_cols
}

# Grouped bar chart setup
x = np.arange(len(labels))  # the label locations
width = 0.18  # width of each bar
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # color for each sim column

plt.figure(figsize=(12, 7))

for i, col in enumerate(sim_cols):
    plt.bar(x + i * width, bin_counts[col].values, width, label=col, color=colors[i])

plt.xlabel("Similarity Range")
plt.ylabel("Count")
plt.title("Binned Similarity Counts for All Columns")
plt.xticks(x + width * 1.5, labels)
plt.legend(title="Similarity Columns")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("similarity_barchart.png")  # Save the figure
plt.show()
