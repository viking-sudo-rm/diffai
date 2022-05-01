from turtle import color
import matplotlib.pyplot as plt
import json
import numpy as np


cornflowerblue = 0x6495ed
blue = 0x0000ff
salmon = 0xfa8072
red = 0xff0000


def get_runs(filename):
    with open(filename, "r") as fp:
        runs = json.load(fp)
    print(f"{filename} has", len(runs), "runs")
    return runs

runs1 = get_runs("runs-.01.json")
runs2 = get_runs("runs-.02.json")

mins = {}
maxs = {}

for runs, eps in [(runs1, .01), (runs2, .02)]:

    max_k = max(run["n_splits"] for run in runs)
    runs_by_n_splits = {k: [run for run in runs if run["n_splits"] == k] for k in range(max_k + 1)}

    # Plot loss as function of k.
    ks, values = zip(*[(k, [run["max_loss"] for run in r]) for k, r in runs_by_n_splits.items()])
    min_values = [min(l) for l in values]
    max_values = [max(l) for l in values]
    plt.figure()
    plt.fill_between(ks, min_values, max_values, alpha=.5, color="cornflowerblue", label="range")
    plt.plot(ks, min_values, marker=".", color="blue", label="min")
    plt.xlabel("Number of splits")
    plt.ylabel("Max loss over ball")
    plt.xlabel(Rf"Max loss vs. number of splits with $\epsilon = {eps}$")
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"figs/loss-{eps}.pdf")
    mins["loss", eps] = min_values
    maxs["loss", eps] = max_values

    # Plot proved as function of k.
    ks, values = zip(*[(k, [run["proved"] for run in runs]) for k, runs in runs_by_n_splits.items()])
    min_values = [min(l) for l in values]
    max_values = [max(l) for l in values]
    plt.figure()
    plt.fill_between(ks, min_values, max_values, alpha=.5, color="cornflowerblue", label="range")
    plt.plot(ks, max_values, marker=".", color="blue", label="max")
    plt.xlabel("Number of splits")
    plt.ylabel("Inputs proved (%)")
    plt.xlabel(Rf"Inputs proved vs. number of splits with $\epsilon = {eps}$")
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"figs/proved-{eps}.pdf")
    mins["proved", eps] = min_values
    maxs["proved", eps] = max_values

runs_by_n_splits1 = {k: [run for run in runs1 if run["n_splits"] == k] for k in range(max_k + 1)}
runs_by_n_splits2 = {k: [run for run in runs2 if run["n_splits"] == k] for k in range(max_k + 1)}

# Plot loss.
plt.figure()
plt.fill_between(ks, mins["loss", .01], maxs["loss", .01], alpha=.5, color="cornflowerblue")
plt.fill_between(ks, mins["loss", .02], maxs["loss", .02], alpha=.5, color="salmon")
plt.plot(ks, mins["loss", .01], marker=".", label=R"$\epsilon = 0.01$", color="blue")
plt.plot(ks, mins["loss", .02], marker=".", label=R"$\epsilon = 0.02$", color="red")
plt.xlabel("Number of splits")
plt.ylabel("Max loss over ball")
plt.tight_layout()
plt.legend()
plt.savefig(f"figs/loss.pdf")

# Plot % proved.
plt.figure()
plt.fill_between(ks, mins["proved", .01], maxs["proved", .01], alpha=.5, color="cornflowerblue")
plt.fill_between(ks, mins["proved", .02], maxs["proved", .02], alpha=.5, color="salmon")
plt.plot(ks, mins["proved", .01], marker=".", label=R"$\epsilon = 0.01$", color="blue")
plt.plot(ks, mins["proved", .02], marker=".", label=R"$\epsilon = 0.02$", color="red")
plt.xlabel("Number of splits")
plt.ylabel("Inputs proved (%)")
plt.tight_layout()
plt.legend()
plt.savefig(f"figs/proved.pdf")

# Plot runtime.
ks, means1 = zip(*[(k, np.mean([run["time"] for run in r])) for k, r in runs_by_n_splits1.items()])
ks, means2 = zip(*[(k, np.mean([run["time"] for run in r])) for k, r in runs_by_n_splits2.items()])
plt.figure()
plt.plot(ks, means1, marker=".", label=R"$\epsilon = 0.01$", color="blue")
plt.plot(ks, means2, marker=".", label=R"$\epsilon = 0.02$", color="red")
plt.xlabel("Number of splits")
plt.ylabel("Runtime (s)")
plt.tight_layout()
plt.legend()
plt.savefig(f"figs/time.pdf")