import json
import matplotlib.pyplot as plt
from collections import Counter

# Load error distribution data
with open('results/error_distribution.json', 'r') as f:
    error_data = json.load(f)

# Extract predicted vs actual labels
predicted_labels = [entry['predicted'] for entry in error_data]
actual_labels = [entry['actual'] for entry in error_data]
error_pairs = [(a, p) for a, p in zip(actual_labels, predicted_labels)]
error_counts = Counter(error_pairs)

labels = [f"{a} → {p}" for a, p in error_counts.keys()]  # Format as 'Actual → Predicted'
counts = list(error_counts.values())

plt.figure(figsize=(10, 5))
plt.bar(labels, counts, color='red')
plt.xlabel("Actual → Predicted")
plt.ylabel("Count")
plt.title("Error Distribution in RNN Model")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig("RNN_results/error_distribution_bar_chart.png")
plt.show()