import matplotlib.pyplot as plt
import json
import numpy as np


cornflowerblue = 0x6495ed
blue = 0x0000ff


with open("runs.json", "r") as fp:
    runs = json.load(fp)

print("# runs", len(runs))

max_k = 6
runs_by_n_splits = {k: [run for run in runs if run["n_splits"] == k] for k in range(max_k)}

# Plot loss as function of k.
ks, values = zip(*[(k, [run["max_loss"] for run in runs]) for k, runs in runs_by_n_splits.items()])
min_values = [min(l) for l in values]
max_values = [max(l) for l in values]
plt.figure()
plt.fill_between(ks, min_values, max_values, alpha=.5, color="cornflowerblue", label="range")
plt.plot(ks, min_values, marker=".", color="blue", label="min")
plt.xlabel("Number of splits")
plt.ylabel("Max loss over ball")
plt.tight_layout()
plt.legend()
plt.show()

# Plot proved as function of k.
ks, values = zip(*[(k, [run["proved"] for run in runs]) for k, runs in runs_by_n_splits.items()])
min_values = [min(l) for l in values]
max_values = [max(l) for l in values]
plt.figure()
plt.fill_between(ks, min_values, max_values, alpha=.5, color="cornflowerblue", label="range")
plt.plot(ks, max_values, marker=".", color="blue", label="max")
plt.xlabel("Number of splits")
plt.ylabel("% proved")
plt.tight_layout()
plt.legend()
plt.show()

# Plot correct proved as function of k.
ks, values = zip(*[(k, [run["correct_proved"] for run in runs]) for k, runs in runs_by_n_splits.items()])
min_values = [min(l) for l in values]
max_values = [max(l) for l in values]
plt.figure()
plt.fill_between(ks, min_values, max_values, alpha=.5, color="cornflowerblue", label="range")
plt.plot(ks, max_values, marker=".", color="blue", label="max")
plt.xlabel("Number of splits")
plt.ylabel("% proved and correct")
plt.tight_layout()
plt.legend()
plt.show()

# Plot correct proved as function of k.
ks, values = zip(*[(k, [run["time"] for run in runs]) for k, runs in runs_by_n_splits.items()])
min_values = [min(l) for l in values]
max_values = [max(l) for l in values]
mean_values = [np.mean(l) for l in values]
plt.figure()
plt.fill_between(ks, min_values, max_values, alpha=.5, color="cornflowerblue", label="range")
plt.plot(ks, mean_values, marker=".", color="blue", label="mean")
plt.xlabel("Number of splits")
plt.ylabel("Runtime (s)")
plt.tight_layout()
plt.legend()
plt.show()